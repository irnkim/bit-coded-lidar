"""Hardware code-distribution delay, in time bins, drawn fresh for every
(pixel, cycle). Models a fast digital clock (distinct from the laser rep
rate) distributing the per-bin code waveform to each pixel: the code value a
pixel's hardware samples may lag the true bin by a small, independent number
of that clock's own cycles -- not a fixed per-pixel property.

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


def describe(probs):
    """One-line summary for run logs, e.g. 'hw_delay=0:0.70,1:0.20,2:0.10'."""
    return 'hw_delay=' + ','.join(f'{v}:{probs[v]:.2f}' for v in sorted(probs))
