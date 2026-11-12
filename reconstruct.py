"""Final depth maps, and the only module that reads ground truth.

Three depth maps come out of every run:

  raw            argmax of the uncoded photon counts -- transient baseline
  reconstructed  argmax of the coded evidence
  bilateral      argmax after cross-bilateral filtering of the coded
                 evidence, guided by the photon-count intensity image

Ground truth is loaded here and nowhere else, only after the batch loop has
finished -- a partial or interrupted run never calls into this module, so it
cannot have seen GT.
"""
from xp_backend import cp
import numpy as np

import dataset_config as dc
import load_scandata as ls

BILATERAL_RADIUS = 5
BILATERAL_SIGMA_SPATIAL = 2.0
BILATERAL_SIGMA_RANGE = 0.15

ADAPTIVE_SIGMA_MIN = 0.8
ADAPTIVE_SIGMA_MAX = 4.0
ADAPTIVE_VAR_WINDOW = 5
ADAPTIVE_SIGMA_RANGE = 0.15


def ground_truth_path(scene_cfg):
    return f'{dc.scene_sim_dir(scene_cfg)}/depth_gt.npz'


def load_ground_truth(scene_cfg):
    """(R, Cw) float array of true depth in cm; may contain NaN where the
    scene has no surface (use nanmean for MAE).
    """
    if scene_cfg.kind == 'scandata':
        return ls.load_gt_depth_cm(scene_cfg)
    return np.load(ground_truth_path(scene_cfg))['depth'][-1]


def guide_image(raw_counts, R, Cw, RES_T):
    """(R, Cw) float64 device array in [0, 1]: per-pixel brightness, summed
    over time from the uncoded histogram, used to guide the bilateral filter.
    """
    totals = raw_counts.reshape(R, Cw, RES_T).sum(axis=-1).astype(cp.float64)
    peak = float(totals.max())
    return totals / peak if peak > 0 else totals


def cross_bilateral_2d(hist_band, guide_band, center_lo, center_hi, radius=BILATERAL_RADIUS,
                        sigma_spatial=BILATERAL_SIGMA_SPATIAL, sigma_range=BILATERAL_SIGMA_RANGE):
    """Cross-bilateral filter over a histogram, guided by an intensity image.
    For each row in [center_lo, center_hi), pools neighboring pixels'
    histograms within radius, weighted by spatial distance and guide-image
    similarity, then argmaxes the pooled histogram.

    Args:
        hist_band:  (band_R, Cw, RES_T) -- row band including context rows.
        guide_band: (band_R, Cw) -- per-pixel intensity guide, same band.
        center_lo, center_hi: row range (within hist_band) to filter.

    Returns:
        (center_hi - center_lo, Cw) float32 -- filtered argmax bin per pixel.
    """
    band_R, Cw, RES_T = hist_band.shape
    results = []
    row_idx = cp.arange(Cw)
    for r in range(center_lo, center_hi):
        acc = cp.zeros((Cw, RES_T), dtype=cp.float64)
        wsum = cp.zeros(Cw, dtype=cp.float64)
        for dr in range(-radius, radius + 1):
            rr = r + dr
            if rr < 0 or rr >= band_R:
                continue
            spatial_r = np.exp(-(dr ** 2) / (2 * sigma_spatial ** 2))
            for dc_ in range(-radius, radius + 1):
                cc = cp.clip(row_idx + dc_, 0, Cw - 1)
                spatial_w = spatial_r * np.exp(-(dc_ ** 2) / (2 * sigma_spatial ** 2))
                range_w = cp.exp(-((guide_band[rr][cc] - guide_band[r]) ** 2) / (2 * sigma_range ** 2))
                w = spatial_w * range_w
                acc += w[:, None] * hist_band[rr][cc]
                wsum += w
        filtered_hist = acc / wsum[:, None]
        results.append(filtered_hist.argmax(axis=-1).astype(cp.float32))
    return cp.stack(results, axis=0)


def cross_bilateral_2d_histogram(hist_band, guide_band, center_lo, center_hi, radius=BILATERAL_RADIUS,
                                  sigma_spatial=BILATERAL_SIGMA_SPATIAL, sigma_range=BILATERAL_SIGMA_RANGE):
    """Like cross_bilateral_2d, but returns the filtered HISTOGRAM itself
    (not its argmax). Needed when the filtered evidence is meant to be fed
    back into an accumulator for further accumulation (cumulative/
    recursive filtering across checkpoints), rather than only read out
    once as a depth estimate.
    """
    band_R, Cw, RES_T = hist_band.shape
    results = []
    row_idx = cp.arange(Cw)
    for r in range(center_lo, center_hi):
        acc = cp.zeros((Cw, RES_T), dtype=cp.float64)
        wsum = cp.zeros(Cw, dtype=cp.float64)
        for dr in range(-radius, radius + 1):
            rr = r + dr
            if rr < 0 or rr >= band_R:
                continue
            spatial_r = np.exp(-(dr ** 2) / (2 * sigma_spatial ** 2))
            for dc_ in range(-radius, radius + 1):
                cc = cp.clip(row_idx + dc_, 0, Cw - 1)
                spatial_w = spatial_r * np.exp(-(dc_ ** 2) / (2 * sigma_spatial ** 2))
                range_w = cp.exp(-((guide_band[rr][cc] - guide_band[r]) ** 2) / (2 * sigma_range ** 2))
                w = spatial_w * range_w
                acc += w[:, None] * hist_band[rr][cc]
                wsum += w
        results.append((acc / wsum[:, None]).astype(cp.float32))
    return cp.stack(results, axis=0)


def box_blur(x, window):
    """(R, Cw) mean of x within a window x window neighborhood, edge-padded.
    Same shift-and-accumulate pattern as local_variance/cross_bilateral_2d
    (avoids cupyx.scipy.ndimage's nvrtc compile issue on this cluster)."""
    R, Cw = x.shape
    pad = window // 2
    row_idx = cp.arange(Cw)
    s = cp.zeros((R, Cw), dtype=cp.float64)
    n = 0
    for dr in range(-pad, pad + 1):
        rows = cp.clip(cp.arange(R) + dr, 0, R - 1)
        xr = x[rows]
        for dc_ in range(-pad, pad + 1):
            cols = cp.clip(row_idx + dc_, 0, Cw - 1)
            s += xr[:, cols]
            n += 1
    return s / n


def local_variance(x, window=ADAPTIVE_VAR_WINDOW):
    """(R, Cw) variance of x within a window x window neighborhood (box
    filter via mean-of-square minus square-of-mean), edge-padded at the
    image border. Uses plain indexed shifts rather than
    cupyx.scipy.ndimage -- that module's elementwise-kernel JIT path hits
    an nvrtc/cuda_fp8.hpp compile error on this cluster's CUDA toolkit,
    while the same shift-and-accumulate pattern cross_bilateral_2d
    already uses below compiles fine.
    """
    R, Cw = x.shape
    pad = window // 2
    row_idx = cp.arange(Cw)
    s = cp.zeros((R, Cw), dtype=cp.float64)
    s2 = cp.zeros((R, Cw), dtype=cp.float64)
    n = 0
    for dr in range(-pad, pad + 1):
        rows = cp.clip(cp.arange(R) + dr, 0, R - 1)
        xr = x[rows]
        for dc_ in range(-pad, pad + 1):
            cols = cp.clip(row_idx + dc_, 0, Cw - 1)
            xv = xr[:, cols]
            s += xv
            s2 += xv * xv
            n += 1
    mean = s / n
    mean_sq = s2 / n
    return cp.clip(mean_sq - mean * mean, 0, None)


def peak_confidence(hist):
    """(R, Cw) in [0, 1]: (top bin - runner-up bin) / (top bin - min bin)
    per pixel, from the coded-evidence histogram. Unlike total photon
    count (near-uniform under uniform illumination, see guide_image),
    this varies with the SHAPE of the evidence -- sharp single-peak
    histograms (confident, usually structured regions) score near 1;
    flat/ambiguous histograms (noisy, usually flat regions) score near 0.
    """
    top1 = hist.max(axis=-1)
    runner_up = cp.where(hist == top1[..., None], -cp.inf, hist).max(axis=-1)
    bottom = hist.min(axis=-1)
    return (top1 - runner_up) / (top1 - bottom + 1e-12)


def adaptive_sigma_spatial(reconstructed_bins, sigma_min=ADAPTIVE_SIGMA_MIN,
                            sigma_max=ADAPTIVE_SIGMA_MAX, window=ADAPTIVE_VAR_WINDOW):
    """(R, Cw) per-pixel spatial sigma: high local variance in the
    pre-filter reconstructed depth (edges/fine structure) -> small sigma
    (narrow, detail-preserving); low local variance (flat/noisy) -> large
    sigma (wide, more averaging). Normalized against this image's own
    variance distribution (90th percentile) rather than a fixed scale, so
    it adapts across scenes/configs without retuning.
    """
    var = local_variance(reconstructed_bins, window=window)
    scale = float(cp.percentile(var, 90)) + 1e-9
    norm_var = cp.clip(var / scale, 0, 1)
    return sigma_max - (sigma_max - sigma_min) * norm_var


def adaptive_sigma_spatial_multiscale(reconstructed_bins, sigma_min=ADAPTIVE_SIGMA_MIN,
                                       sigma_max=ADAPTIVE_SIGMA_MAX,
                                       windows=(5, 13)):
    """Like adaptive_sigma_spatial, but combines local variance at several
    window sizes instead of one. A small window alone is fooled by a large
    coding group's block-uniform pre-filter depth (near-zero variance
    INSIDE the block looks "flat" even next to a real edge at the block
    boundary); taking the max of each scale's own-normalized variance
    means a pixel only gets the large (over-smoothing) sigma when NO
    scale finds structure nearby, instead of trusting the smallest scale
    alone.
    """
    norm_vars = []
    for window in windows:
        var = local_variance(reconstructed_bins, window=window)
        scale = float(cp.percentile(var, 90)) + 1e-9
        norm_vars.append(cp.clip(var / scale, 0, 1))
    norm_var = cp.stack(norm_vars, axis=0).max(axis=0)
    return sigma_max - (sigma_max - sigma_min) * norm_var


def cross_bilateral_2d_adaptive(hist_band, guide_band, sigma_spatial_band,
                                 center_lo, center_hi, radius=BILATERAL_RADIUS,
                                 sigma_range=ADAPTIVE_SIGMA_RANGE):
    """Like cross_bilateral_2d, but sigma_spatial varies per center pixel
    (sigma_spatial_band: (band_R, Cw), same band as hist_band/guide_band)
    instead of being a single scalar for the whole image. radius stays
    fixed (the loop bound) -- adaptivity comes from how fast the Gaussian
    weight decays within that fixed window, which is the vectorizable
    equivalent of a variable radius.
    """
    band_R, Cw, RES_T = hist_band.shape
    results = []
    row_idx = cp.arange(Cw)
    for r in range(center_lo, center_hi):
        sigma_row = sigma_spatial_band[r]  # (Cw,) -- one sigma per center pixel
        acc = cp.zeros((Cw, RES_T), dtype=cp.float64)
        wsum = cp.zeros(Cw, dtype=cp.float64)
        for dr in range(-radius, radius + 1):
            rr = r + dr
            if rr < 0 or rr >= band_R:
                continue
            spatial_r = cp.exp(-(dr ** 2) / (2 * sigma_row ** 2))
            for dc_ in range(-radius, radius + 1):
                cc = cp.clip(row_idx + dc_, 0, Cw - 1)
                spatial_w = spatial_r * cp.exp(-(dc_ ** 2) / (2 * sigma_row ** 2))
                range_w = cp.exp(-((guide_band[rr][cc] - guide_band[r]) ** 2) / (2 * sigma_range ** 2))
                w = spatial_w * range_w
                acc += w[:, None] * hist_band[rr][cc]
                wsum += w
        filtered_hist = acc / wsum[:, None]
        results.append(filtered_hist.argmax(axis=-1).astype(cp.float32))
    return cp.stack(results, axis=0)


def depth_maps_adaptive(accumulator, raw_counts, scene_cfg, reconstructed_bins,
                         radius=BILATERAL_RADIUS, sigma_min=ADAPTIVE_SIGMA_MIN,
                         sigma_max=ADAPTIVE_SIGMA_MAX, var_window=ADAPTIVE_VAR_WINDOW,
                         sigma_range=ADAPTIVE_SIGMA_RANGE):
    """Adaptive counterpart of depth_maps()'s 'bilateral' map: per-pixel
    spatial sigma from local depth-variance (adaptive_sigma_spatial), and
    the guide image swapped from raw brightness to peak_confidence (see
    its docstring -- brightness is near-uniform under uniform
    illumination and carries no information for the range term).
    reconstructed_bins: (R, Cw) pre-filter argmax depth bins, as returned
    by depth_maps()['reconstructed'] but in BIN units, not cm -- pass
    accumulator.argmax(axis=-1) reshaped, not the cm-converted map.
    Returns (R, Cw) float32 filtered argmax bins (bin units, not cm).
    """
    R, Cw, RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T
    sigma_map = adaptive_sigma_spatial(
        cp.asarray(reconstructed_bins), sigma_min=sigma_min, sigma_max=sigma_max, window=var_window)
    confidence_guide = peak_confidence(accumulator.reshape(R, Cw, RES_T))
    filtered_bins = cross_bilateral_2d_adaptive(
        accumulator.reshape(R, Cw, RES_T), confidence_guide, sigma_map,
        0, R, radius=radius, sigma_range=sigma_range)
    return filtered_bins


def bins_to_cm(bins, per_bin_cm):
    """Bin indices to depth in cm, at the bin center (+0.5 removes a
    constant half-bin bias)."""
    return (np.asarray(bins).astype(np.float32) + 0.5) * per_bin_cm


def depth_maps(accumulator, raw_counts, scene_cfg,
               radius=BILATERAL_RADIUS,
               sigma_spatial=BILATERAL_SIGMA_SPATIAL,
               sigma_range=BILATERAL_SIGMA_RANGE):
    """The three depth maps in cm (see module docstring): dict with 'raw',
    'reconstructed', 'bilateral', each an (R, Cw) float32 array.
    """
    R, Cw, RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T
    per_bin_cm = scene_cfg.PER_BIN_CM

    raw = bins_to_cm(
        cp.asnumpy(raw_counts.reshape(R, Cw, RES_T).argmax(axis=-1)), per_bin_cm)

    reconstructed = bins_to_cm(
        cp.asnumpy(accumulator.argmax(axis=-1)).reshape(R, Cw), per_bin_cm)

    # Filtering the full histogram and then taking argmax borrows evidence
    # from neighbors; smoothing the depth map afterwards would just average
    # already-wrong answers.
    filtered_bins = cross_bilateral_2d(
        accumulator.reshape(R, Cw, RES_T), guide_image(raw_counts, R, Cw, RES_T),
        0, R, radius=radius, sigma_spatial=sigma_spatial, sigma_range=sigma_range)
    bilateral = bins_to_cm(cp.asnumpy(filtered_bins), per_bin_cm)
    return dict(raw=raw, reconstructed=reconstructed, bilateral=bilateral)


def mae(estimate, gt):
    """Mean absolute depth error in cm (nanmean: gt may contain NaN where
    the scene has no surface)."""
    return float(np.nanmean(np.abs(np.asarray(estimate) - np.asarray(gt))))


def evaluate(accumulator, raw_counts, scene_cfg, **bilateral_kwargs):
    """Depth maps plus their MAEs against ground truth. Call only after
    acquisition is complete; it reads ground truth.

    Returns:
        maps:   dict of three depth maps, as depth_maps().
        gt:     (R, Cw) ground-truth depth in cm.
        errors: dict with the same three keys, each a float MAE in cm.
    """
    maps = depth_maps(accumulator, raw_counts, scene_cfg, **bilateral_kwargs)
    gt = load_ground_truth(scene_cfg)
    errors = {name: mae(depth, gt) for name, depth in maps.items()}
    return maps, gt, errors


def save(path, maps, gt, errors, metadata):
    """Write depth maps, ground truth, MAEs and run configuration to path
    (.npz). metadata values are wrapped with np.array so scalars and
    strings both survive the round trip.
    """
    payload = dict(gt=gt, **{f'depth_{name}': depth for name, depth in maps.items()},
                   **{f'mae_{name}': np.float32(value) for name, value in errors.items()})
    for key, value in metadata.items():
        payload[key] = np.array(value)
    np.savez(path, **payload)
    return path
