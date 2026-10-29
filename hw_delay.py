"""Hardware code-distribution delay, in time bins. Models a fast digital
clock (distinct from the laser rep rate) distributing the per-bin code
waveform to each pixel: the code value a pixel's hardware samples may lag
the true bin by a small number of that clock's own cycles.

Two granularities:
    draw(n_pixels, batch_size)                       independent per
        (pixel, cycle) -- the default, no shared clock-domain structure.
    draw_per_group(labels_flat, n_groups, batch_size) one draw per
        (batch, group): every pixel in the same spatial group, and every
        cycle in the batch, shares the identical delay -- models all
        pixels in a group hanging off the same clock-distribution branch.

Usage:
    delay_source = DelaySource(probs={0: 0.7, 1: 0.2, 2: 0.1})
    delay_bins = delay_source.draw(n_pixels, batch_size)   # (n_pixels, batch_size)
    bits = encode.group_bits(code_pb, labels_flat, n_groups, arrivals,
                              pixel_delay_bins=delay_bins)

``probs`` is a placeholder until real clock-frequency figures are known --
override via HW_DELAY_PROBS in run_nonadaptive.py.
"""
import cupy as cp
import numpy as np

GLOBAL_SEED = 4096  # same constant as codes.py / bernoulli_coding.py / dataset_config.py
SEED_SALT = 777      # offsets this module's stream from codes.py's own GLOBAL_SEED + row_lo


class DelaySource:
    """Draws a fresh (n_pixels, batch_size) delay-in-bins array every batch.

    Args:
        probs:  dict[int, float] -- delay value (bins) -> population
                fraction, e.g. {0: 0.7, 1: 0.2, 2: 0.1}. Must sum to 1.
        row_lo: row-band offset, same role as CodeSource's.
    """

    def __init__(self, probs, row_lo=0):
        values = np.array(sorted(probs), dtype=np.int32)
        weights = np.array([probs[v] for v in values], dtype=np.float64)
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f'probs must sum to 1, got {weights.sum()}')
        self.probs = probs
        self.values = cp.asarray(values)
        # Ascending cumulative thresholds; draw(u) buckets by how many
        # thresholds u has cleared. cum[-1] == 1.0 exactly by construction
        # (not just np.isclose), so u in [0,1) always lands in a bucket.
        self.cum = cp.asarray(np.cumsum(weights))
        self.rng = cp.random.default_rng(GLOBAL_SEED + SEED_SALT + row_lo)

    def draw(self, n_pixels, batch_size):
        """This batch's (n_pixels, batch_size) delay-in-bins array. Every
        call advances the generator, so resume needs fast_forward, not a seek.
        """
        u = self.rng.random((n_pixels, batch_size), dtype=cp.float32)
        bucket = cp.zeros((n_pixels, batch_size), dtype=cp.int32)
        for threshold in self.cum[:-1]:
            bucket += (u >= threshold).astype(cp.int32)
        return self.values[bucket]

    def fast_forward(self, n_batches, n_pixels, batch_size):
        """Replay (and discard) n_batches draws so a resumed run's generator
        lands in the right position."""
        for _ in range(n_batches):
            self.draw(n_pixels, batch_size)

    def draw_per_group(self, labels_flat, n_groups, batch_size):
        """One delay draw per (batch, group) instead of per (pixel, cycle):
        every pixel in the same spatial group, and every cycle in this
        batch, shares the identical delay value. Returns (n_pixels,
        batch_size) -- a broadcast view, not a materialized copy -- so it's
        a drop-in replacement for draw()'s output shape wherever it's only
        read (encode.group_bits indexes it, never writes it).
        """
        u = self.rng.random((n_groups,), dtype=cp.float32)
        bucket = cp.zeros((n_groups,), dtype=cp.int32)
        for threshold in self.cum[:-1]:
            bucket += (u >= threshold).astype(cp.int32)
        group_delay = self.values[bucket]       # (n_groups,)
        pixel_delay = group_delay[labels_flat]  # (n_pixels,)
        return cp.broadcast_to(pixel_delay[:, None], (pixel_delay.shape[0], batch_size))

    def fast_forward_per_group(self, n_batches, n_groups):
        """Replay (and discard) n_batches per-group draws, for resuming a
        draw_per_group run (see fast_forward)."""
        for _ in range(n_batches):
            u = self.rng.random((n_groups,), dtype=cp.float32)
            del u


def describe(probs):
    """One-line summary for run logs, e.g. 'hw_delay=0:0.70,1:0.20,2:0.10'."""
    return 'hw_delay=' + ','.join(f'{v}:{probs[v]:.2f}' for v in sorted(probs))
