"""Bernoulli(p=0.5) coding function via genuine RNG sampling
(cupy.random.Generator.binomial), one dense (row_band, Cw, RES_T) tensor per
batch, discarded after use -- bounds memory regardless of scene size.

cupy's generator is deterministic given (seed, exact sequence of call sizes);
it is not safe to split one call into differently-sized calls and expect a
slice of the single-call result. Resume is replay-based (the generator can't
be pickled or seeked): recreate with the same seed and draw-and-discard the
same per-batch shapes up to the resume point.
"""
import cupy as cp

GLOBAL_SEED = 4096


def make_generator(row_lo, seed=GLOBAL_SEED):
    """Row-band-scoped generator: each band gets its own independent seed."""
    return cp.random.default_rng(seed + row_lo)


def draw_pulse_coding(rng, row_band, Cw, res_t, p=0.5):
    """(row_band, Cw, res_t) int8 (0/1) Bernoulli(p) draw. Must be called
    with the same shape every time for a given rng to stay reproducible.
    """
    return rng.binomial(1, p, size=(row_band, Cw, res_t)).astype(cp.int8)


def fast_forward(rng, n_pulses_to_skip, row_band, Cw, res_t, p=0.5):
    """Replay (draw-and-discard) n_pulses_to_skip draws on a fresh rng, so
    it lands where an uninterrupted run would be after that many pulses.
    """
    for _ in range(n_pulses_to_skip):
        draw_pulse_coding(rng, row_band, Cw, res_t, p)
