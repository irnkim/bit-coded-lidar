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
import cupy as cp
import numpy as np

import dataset_config as dc
import load_scandata as ls

BILATERAL_RADIUS = 5
BILATERAL_SIGMA_SPATIAL = 2.0
BILATERAL_SIGMA_RANGE = 0.15


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
