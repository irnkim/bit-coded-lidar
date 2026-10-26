"""Loader for the real hardware "scan data" scenes (face_scanning, deer).
No independent GT depth exists for this data; their measured histogram is
used as the transient/probability-distribution input to photon sampling --
fresh synthetic events are resampled from the histogram's shape, the real
detected photons are not replayed. Ground truth is the argmax of that same
measured histogram.
"""

import numpy as np


def load_transient(scene_cfg):
    """(R, Cw, RES_T) float32 -- the scene's raw measured histogram, sliced
    to (scene_cfg.R, scene_cfg.Cw) (a no-op for full-size scenes; lets a
    cropped variant, e.g. deer_adaptive, reuse the same file on disk).
    """
    transient = np.load(scene_cfg.transient_path).astype(np.float32)
    return transient[:scene_cfg.R, :scene_cfg.Cw]


def load_gt_depth_cm(scene_cfg):
    """(R, Cw) float64 in cm -- argmax of the scene's own measured
    histogram (no independent ground truth exists for this data).
    """
    transient = load_transient(scene_cfg)
    argmax_bin = transient.argmax(axis=-1)
    return (argmax_bin.astype(np.float64) + 0.5) * scene_cfg.PER_BIN_CM


def load_valid_signal_mask(scene_cfg):
    """(R, Cw) bool -- True where the scene has real signal (background/
    no-return pixels excluded), from the scene's nosignal-mask PNG
    (1.0 = no signal).
    """
    import os
    import matplotlib.pyplot as plt
    mask_dir = os.path.dirname(scene_cfg.transient_path)
    mask_path = f'{mask_dir}/nosignal-mask_r-{scene_cfg.R}-c-{scene_cfg.Cw}.png'
    nosignal = plt.imread(mask_path)
    if nosignal.ndim == 3:
        nosignal = nosignal[..., 0]
    return nosignal < 0.5
