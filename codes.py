"""Temporal codes: A in y = Ax. Four per-pulse-draw families: bernoulli,
lowpass, trig, chipoffset. A fresh dense 0/1 tensor is drawn and held for
the whole batch; reproducible from (seed, exact sequence of call shapes) --
there is no way to seek, so resume replays.

``sharing`` sets assignment granularity: 'unshared' is one code per pixel;
'shared' is one code per group, fanned out to its member pixels. Under a
sliding lattice the group count (and so the draw shape) changes between
batches; that stays reproducible because the shape sequence is a
deterministic function of the batch index.

Output is always (R*Cw, RES_T) int8 in {-1, +1}, the form encode.py indexes.
"""
import cupy as cp
import numpy as np

import bernoulli_coding as bc
import jitter_robust_coding as jrc

DRAW_FAMILIES = ('bernoulli', 'lowpass', 'trig', 'chipoffset')
CODE_FAMILIES = DRAW_FAMILIES
SHARING_MODES = ('unshared', 'shared')

# Presentation/display names for the three jitter-robust families (lowpass,
# trig, chipoffset are all literally bernoulli-based: lowpass = smoothed-
# then-thresholded, chipoffset = grouped into fixed-width chips, trig =
# summed cosine harmonics). Accepted as aliases for CODE=/family= so
# running an experiment with either name does the identical thing --
# resolved to the canonical name immediately, before it ever reaches a
# result filename, checkpoint fingerprint, or RNG seed, so this changes
# nothing about already-computed results or in-flight runs.
FAMILY_ALIASES = {
    'wide-bernoulli': 'chipoffset',
    'filtered-bernoulli': 'lowpass',
    'cosine-wave': 'trig',
}


def resolve_family(name):
    """Canonical family name for `name`, passing through aliases (see
    FAMILY_ALIASES) and leaving an already-canonical name untouched."""
    return FAMILY_ALIASES.get(name, name)

GLOBAL_SEED = 4096  # shared with bernoulli_coding, jitter_robust_coding, dataset_config.EVENT_SEED

_DRAW_FNS = {
    'bernoulli': bc.draw_pulse_coding,
    'lowpass': jrc.draw_pulse_coding_lowpass,
    'trig': jrc.draw_pulse_coding_trig,
    'chipoffset': jrc.draw_pulse_coding_chip_offset,
}


class CodeSource:
    """Produces one batch of temporal codes at a time.

    Args:
        scene_cfg: a dataset_config scene (supplies R, Cw, RES_T).
        family:    one of CODE_FAMILIES.
        sharing:   'unshared' (one code per pixel) or 'shared' (one per group).
        row_lo:    row-band offset used to seed the draw families.
    """

    def __init__(self, scene_cfg, family='bernoulli', sharing='unshared', row_lo=0):
        if family not in CODE_FAMILIES:
            raise ValueError(f'family must be one of {CODE_FAMILIES}, got {family!r}')
        if sharing not in SHARING_MODES:
            raise ValueError(f'sharing must be one of {SHARING_MODES}, got {sharing!r}')
        self.scene_cfg = scene_cfg
        self.family = family
        self.sharing = sharing
        self.R, self.Cw, self.RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T
        self.rng = (jrc.make_generator(row_lo) if family != 'bernoulli'
                    else bc.make_generator(row_lo))

    def _draw_unshared(self):
        """One independent code per pixel: (R*Cw, RES_T) int8 in {-1, +1}."""
        code01 = _DRAW_FNS[self.family](self.rng, self.R, self.Cw, self.RES_T)
        return (code01.reshape(self.R * self.Cw, self.RES_T).astype(cp.int8) * 2 - 1)

    def _draw_shared(self, labels_flat, n_groups):
        """One code per group, fanned out to its pixels: (R*Cw, RES_T) int8 in {-1, +1}."""
        code01 = _DRAW_FNS[self.family](self.rng, n_groups, 1, self.RES_T)
        group_code = (code01.reshape(n_groups, self.RES_T).astype(cp.int8) * 2 - 1)
        return group_code[labels_flat]

    def draw(self, batch_index, labels_flat=None, n_groups=None):
        """This batch's temporal codes: (R*Cw, RES_T) int8 in {-1, +1}, held
        constant across all B cycles. Pass the SAME array to both
        encode.group_bits and decode.backproject. batch_index must advance
        by one each call (the generator is stateful); labels_flat/n_groups
        are required in shared mode.
        """
        if self.sharing == 'shared' and (labels_flat is None or n_groups is None):
            raise ValueError(f'{self.family} in shared mode needs labels_flat and n_groups')
        if self.sharing == 'shared':
            return self._draw_shared(labels_flat, n_groups)
        return self._draw_unshared()

    def fast_forward(self, n_batches, labels_for_batch=None):
        """Replay n_batches draws (discarded) so a resumed run's generator
        lands in the right position. labels_for_batch(batch) -> (labels_flat,
        n_groups) is required in shared mode, since the draw shape depends
        on each batch's group count.
        """
        for batch in range(n_batches):
            if self.sharing == 'shared':
                if labels_for_batch is None:
                    raise ValueError('shared-mode replay needs labels_for_batch')
                labels_flat, n_groups = labels_for_batch(batch)
                self._draw_shared(labels_flat, n_groups)
            else:
                self._draw_unshared()

    def describe(self):
        return f'code={self.family} sharing={self.sharing}'
