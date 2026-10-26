"""Three jitter-robust code generators, each a drop-in replacement for
bernoulli_coding.draw_pulse_coding: same call signature, same (row_band, Cw,
res_t) int8 0/1 output, same per-pulse fresh-draw contract.

Same reproducibility model as bernoulli_coding.py: same seed + same exact
sequence of call shapes reproduces bit-identical results; resume is
replay-based via fast_forward.
"""
import cupy as cp

GLOBAL_SEED = 4096


def make_generator(row_lo, seed=GLOBAL_SEED):
    return cp.random.default_rng(seed + row_lo)


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


def draw_pulse_coding_trig(rng, row_band, Cw, res_t):
    """Fresh random amplitude+phase per harmonic, per pixel, every call.
    Thresholds at the realization's own median (not zero) to force an
    exact balanced 0/1 split regardless of the draw.
    """
    t = cp.arange(res_t, dtype=cp.float32)
    amps = rng.standard_normal(size=(row_band, Cw, TRIG_K_HARM, 1), dtype=cp.float32)
    phases = rng.uniform(0, 2 * cp.pi, size=(row_band, Cw, TRIG_K_HARM, 1)).astype(cp.float32)
    harmonics = cp.arange(1, TRIG_K_HARM + 1, dtype=cp.float32).reshape(1, 1, TRIG_K_HARM, 1)
    waves = amps * cp.cos(2 * cp.pi * harmonics * t.reshape(1, 1, 1, res_t) / res_t + phases)
    wave = waves.sum(axis=2)
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


def fast_forward(draw_fn, rng, n_pulses_to_skip, row_band, Cw, res_t):
    for _ in range(n_pulses_to_skip):
        draw_fn(rng, row_band, Cw, res_t)


# Batched variants: generate one block's entire batch history in a single
# call, shape (n_units, g_h, g_w, RES_T), by reshaping (n_units, g_h*g_w)
# into the (row_band, Cw) slot the per-pulse functions above expect.
_BATCH_DRAW_FNS = {
    'lowpass': draw_pulse_coding_lowpass,
    'trig': draw_pulse_coding_trig,
    'chipoffset': draw_pulse_coding_chip_offset,
}


def draw_batched(code_family, rng, n_units, g_h, g_w, res_t):
    code01 = _BATCH_DRAW_FNS[code_family](rng, n_units, g_h * g_w, res_t)
    return code01.reshape(n_units, g_h, g_w, res_t)
