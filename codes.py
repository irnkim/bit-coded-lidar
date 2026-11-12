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

The three jitter-robust families (lowpass, trig, chipoffset) and plain
bernoulli all live in this one file -- same reproducibility model
throughout: same seed + same exact sequence of call shapes reproduces
bit-identical results; resume is replay-based via fast_forward, since
neither cupy's nor numpy's Generator can be pickled or seeked.

KCodebookSource (below) is a different scheme: K codes predefined ONCE
(not redrawn per batch), shared across the whole scene, with each pixel
independently picking a random index into them every CYCLE.
"""
from xp_backend import cp

GLOBAL_SEED = 4096  # shared with dataset_config.EVENT_SEED

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


def make_generator(row_lo, seed=GLOBAL_SEED):
    """Row-band-scoped generator: each band gets its own independent seed."""
    return cp.random.default_rng(seed + row_lo)


# -- 0. Plain Bernoulli(p=0.5) -------------------------------------------

def draw_pulse_coding(rng, row_band, Cw, res_t, p=0.5):
    """(row_band, Cw, res_t) int8 (0/1) Bernoulli(p) draw. Must be called
    with the same shape every time for a given rng to stay reproducible.
    """
    return rng.binomial(1, p, size=(row_band, Cw, res_t)).astype(cp.int8)


# -- 1. Low-pass filtered Bernoulli --------------------------------------
LOWPASS_KERNEL_SIGMA = 1.2  # bins; kernel width ~jitter scale
LOWPASS_KERNEL_HALFWIDTH = 3


def _lowpass_kernel():
    x = cp.arange(-LOWPASS_KERNEL_HALFWIDTH, LOWPASS_KERNEL_HALFWIDTH + 1)
    k = cp.exp(-0.5 * (x / LOWPASS_KERNEL_SIGMA) ** 2)
    return k / k.sum()


def draw_pulse_coding_lowpass(rng, row_band, Cw, res_t):
    """Gaussian noise per pixel, smoothed along the time-bin axis, then
    thresholded at zero. 0/1 output (bipolar conversion happens downstream).
    Smoothing is an explicit weighted sum of shifted copies, not
    cupyx.scipy.ndimage (its JIT kernel fails to compile in this environment).
    """
    noise = rng.standard_normal(size=(row_band, Cw, res_t), dtype=cp.float32)
    kernel = _lowpass_kernel()
    padded = cp.pad(noise, ((0, 0), (0, 0), (LOWPASS_KERNEL_HALFWIDTH, LOWPASS_KERNEL_HALFWIDTH)), mode='edge')
    smoothed = cp.zeros_like(noise)
    for i, w in enumerate(kernel):
        smoothed += w * padded[:, :, i:i + res_t]
    return (smoothed > 0).astype(cp.int8)


# -- 2. Random low-harmonic trigonometric code ---------------------------
TRIG_K_HARM = 3  # number of low harmonics; period of highest >> jitter scale


TRIG_CHUNK_TARGET_ELEMENTS = 2_500_000_000 // 4  # ~1GB of float32 per chunk's (rows,Cw,k,res_t) tensor


def draw_pulse_coding_trig(rng, row_band, Cw, res_t, k=TRIG_K_HARM):
    """Fresh random amplitude+phase per harmonic, per pixel, every call.
    Thresholds at the realization's own median (not zero) to force an
    exact balanced 0/1 split regardless of the draw. k is the harmonic
    count (frequency, F in the F/K convention); defaults to TRIG_K_HARM.

    The (row_band, Cw, k, res_t) intermediate (amps * cos(...), before
    summing over k) is the memory-dominant step -- at k=50, full-scene
    this is ~30GB, fine in GPU VRAM but far over this cluster's per-user
    15GB CPU-RAM cgroup cap (BACKEND=cpu). Computed in ROW CHUNKS instead
    (same amps/phases draw, same math, bounded peak memory) -- harmless
    on GPU too, so this isn't backend-conditional.
    """
    t = cp.arange(res_t, dtype=cp.float32)
    amps = rng.standard_normal(size=(row_band, Cw, k, 1), dtype=cp.float32)
    phases = rng.uniform(0, 2 * cp.pi, size=(row_band, Cw, k, 1)).astype(cp.float32)
    harmonics = cp.arange(1, k + 1, dtype=cp.float32).reshape(1, 1, k, 1)

    chunk_rows = max(1, TRIG_CHUNK_TARGET_ELEMENTS // (Cw * k * res_t))
    wave = cp.zeros((row_band, Cw, res_t), dtype=cp.float32)
    for lo in range(0, row_band, chunk_rows):
        hi = min(lo + chunk_rows, row_band)
        waves_chunk = amps[lo:hi] * cp.cos(
            2 * cp.pi * harmonics * t.reshape(1, 1, 1, res_t) / res_t + phases[lo:hi])
        wave[lo:hi] = waves_chunk.sum(axis=2)
        del waves_chunk

    median = cp.median(wave, axis=-1, keepdims=True)
    return (wave > median).astype(cp.int8)


# -- 3. Random-offset chip repetition -------------------------------------
CHIP_WIDTH = 3  # bins; matched to expected +-1-bin jitter scale


def draw_pulse_coding_chip_offset(rng, row_band, Cw, res_t):
    """Fresh random offset in [0, CHIP_WIDTH) per pixel per call, then
    independent random bits per chip from that offset.
    """
    k = CHIP_WIDTH
    offsets = rng.integers(0, k, size=(row_band, Cw, 1))
    t = cp.arange(res_t).reshape(1, 1, res_t)
    # chip index per bin per pixel: index 0 is the partial first chip (t < offset)
    chip_idx = cp.where(t < offsets, 0, 1 + (t - offsets) // k)
    n_chips_max = int(res_t // k) + 2
    chip_bits = rng.integers(0, 2, size=(row_band, Cw, n_chips_max)).astype(cp.int8)
    row_i = cp.arange(row_band).reshape(row_band, 1, 1)
    col_i = cp.arange(Cw).reshape(1, Cw, 1)
    return chip_bits[row_i, col_i, chip_idx]


_DRAW_FNS = {
    'bernoulli': draw_pulse_coding,
    'lowpass': draw_pulse_coding_lowpass,
    'trig': draw_pulse_coding_trig,
    'chipoffset': draw_pulse_coding_chip_offset,
}


class CodeSource:
    """Produces one batch of temporal codes at a time.

    Args:
        scene_cfg: a dataset_config scene (supplies R, Cw, RES_T).
        family:    one of CODE_FAMILIES.
        sharing:   'unshared' (one code per pixel) or 'shared' (one per group).
        row_lo:    row-band offset used to seed the draw families.
    """

    def __init__(self, scene_cfg, family='bernoulli', sharing='unshared', row_lo=0, trig_k=None):
        if family not in CODE_FAMILIES:
            raise ValueError(f'family must be one of {CODE_FAMILIES}, got {family!r}')
        if sharing not in SHARING_MODES:
            raise ValueError(f'sharing must be one of {SHARING_MODES}, got {sharing!r}')
        self.scene_cfg = scene_cfg
        self.family = family
        self.sharing = sharing
        self.R, self.Cw, self.RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T
        self.rng = make_generator(row_lo)
        # Harmonic count (frequency, F) for the trig family only; ignored by
        # every other family. None keeps this module's own default.
        self.trig_k = trig_k if trig_k is not None else TRIG_K_HARM

    def _draw(self, row_band, Cw):
        if self.family == 'trig':
            return _DRAW_FNS[self.family](self.rng, row_band, Cw, self.RES_T, k=self.trig_k)
        return _DRAW_FNS[self.family](self.rng, row_band, Cw, self.RES_T)

    def _draw_unshared(self):
        """One independent code per pixel: (R*Cw, RES_T) int8 in {-1, +1}."""
        code01 = self._draw(self.R, self.Cw)
        return (code01.reshape(self.R * self.Cw, self.RES_T).astype(cp.int8) * 2 - 1)

    def _draw_shared(self, labels_flat, n_groups):
        """One code per group, fanned out to its pixels: (R*Cw, RES_T) int8 in {-1, +1}."""
        code01 = self._draw(n_groups, 1)
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
        trig_part = f' trig_k={self.trig_k}' if self.family == 'trig' else ''
        return f'code={self.family} sharing={self.sharing}{trig_part}'


class KCodebookSource:
    """K predefined codes (same family/trig_k as a regular draw), shared
    across the whole scene -- NOT per-pixel. Each pixel, each CYCLE within
    a batch, independently picks a uniformly random index into this fixed
    codebook (with replacement). Unlike CodeSource, the codebook is drawn
    ONCE at construction, not redrawn per batch -- only the per-(pixel,
    cycle) INDEX is fresh every batch.

    draw(batch_size) returns (idx, codebook):
        idx:      (R*Cw, batch_size) int32, each in [0, K) -- which
                  codebook row this pixel used on this cycle.
        codebook: (K, RES_T) int8 in {-1, +1} -- fixed for the whole run.
    encode/decode then index into codebook with idx per (pixel, cycle)
    instead of sharing one code for the whole batch (see encode.group_bits
    and decode.backproject's *_kcodebook variants).
    """

    def __init__(self, scene_cfg, K, family='trig', row_lo=0, trig_k=None, seed=GLOBAL_SEED):
        if family not in CODE_FAMILIES:
            raise ValueError(f'family must be one of {CODE_FAMILIES}, got {family!r}')
        self.scene_cfg = scene_cfg
        self.K = K
        self.family = family
        self.R, self.Cw, self.RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T
        self.trig_k = trig_k if trig_k is not None else TRIG_K_HARM

        codebook_rng = make_generator(row_lo, seed=seed)
        if family == 'trig':
            code01 = draw_pulse_coding_trig(codebook_rng, K, 1, self.RES_T, k=self.trig_k)
        else:
            code01 = _DRAW_FNS[family](codebook_rng, K, 1, self.RES_T)
        self.codebook = (code01.reshape(K, self.RES_T).astype(cp.int8) * 2 - 1)

        # A separate stream from the codebook draw, so changing K doesn't
        # reshuffle which index sequence each pixel/batch gets (and vice
        # versa) -- same row_lo/seed convention as every other source here.
        self.idx_rng = cp.random.default_rng(seed + row_lo + 1)

    def draw(self, batch_size):
        """(R*Cw, batch_size) int32 indices into self.codebook, fresh every
        call -- advances the generator, so resume needs fast_forward."""
        return self.idx_rng.integers(0, self.K, size=(self.R * self.Cw, batch_size))

    def fast_forward(self, n_batches, batch_size):
        for _ in range(n_batches):
            self.draw(batch_size)

    def describe(self):
        return f'code={self.family} K={self.K} trig_k={self.trig_k}'
