"""Per-scene parameters. Each scene carries its own (R, Cw, RES_T,
PER_BIN_CM) rather than a shared constant, since downstream code (sampling,
coding, decode) takes them as explicit arguments.

Physics differs by scene kind:
  - 'flashlidar': a rendered, noise-free ground-truth transient, so it gets
    the full synthetic sensor model (pdp=0.3, app=0.01, ctp=0.001,
    dcr=3000/s, jitter on).
  - 'scandata': a real hardware histogram, already containing real jitter,
    dark/background, and afterpulsing. pdp=1.0 (no thinning), jitter/app/dcr
    off, to avoid doubling noise already baked into the data.
"""

import os
from dataclasses import dataclass, field

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_REPO_ROOT, 'data')
SIM_DIR = os.path.join(_REPO_ROOT, 'data')
RESULTS_DIR = os.path.join(_REPO_ROOT, 'results')

C = 3e8  # speed of light, m/s

M = 500_000
PULSE_CHUNK = 10_000
N_BLOCKS = M // PULSE_CHUNK
N_CKPT = 500
CKPT_INTERVAL = M // N_CKPT
N_ROWS_DIAGNOSTIC = 5
GLOBAL_SEED = 4096
EVENT_SEED = 4096  # separate stream from GLOBAL_SEED: photon sampling vs. coding

KIND_DIR = {'flashlidar': 'rendered_images', 'scandata': 'scan_data'}


def scene_sim_dir(scene_cfg):
    return f'{SIM_DIR}/{KIND_DIR[scene_cfg.kind]}/{scene_cfg.name}'


def scene_results_dir(scene_cfg):
    return f'{RESULTS_DIR}/{KIND_DIR[scene_cfg.kind]}/{scene_cfg.name}'


PHYSICS = {
    'flashlidar': dict(pdp=0.3, app=0.01, ctp=0.001, dcr=3000.0, noise_scale=1.0, use_jitter=True),
    'scandata':   dict(pdp=1.0, app=0.0,  ctp=0.0,   dcr=0.0,    noise_scale=0.0, use_jitter=False),
}


@dataclass
class SceneConfig:
    name: str
    kind: str            # 'flashlidar' or 'scandata'
    R: int                # rows
    Cw: int               # columns (avoid clashing with speed-of-light C)
    RES_T: int
    PER_BIN_CM: float
    transient_path: str
    depth_path: str = None   # only set for flashlidar (real, independent GT)

    @property
    def physics(self):
        return PHYSICS[self.kind]

    @property
    def diagnostic_rows(self):
        import numpy as np
        return sorted(set(np.linspace(0, self.R - 1, N_ROWS_DIAGNOSTIC).round().astype(int).tolist()))


FLASHLIDAR_SCENES = {
    'bathroom-cycles-2': SceneConfig(
        name='bathroom-cycles-2', kind='flashlidar', R=240, Cw=320, RES_T=2000,
        PER_BIN_CM=0.5,
        transient_path=f'{DATA_DIR}/rendered_images/transient_images_240x320_nt-2000/'
                        f'bathroom-cycles-2_nr-240_nc-320_nt-2000_samples-2048_view-0.npz',
        depth_path=f'{DATA_DIR}/rendered_images/depth_images_240x320_nt-2000/'
                   f'bathroom-cycles-2_nr-240_nc-320_nt-2000_samples-2048_view-0.npy',
    ),
    'kitchen-2': SceneConfig(
        name='kitchen-2', kind='flashlidar', R=240, Cw=320, RES_T=2000,
        PER_BIN_CM=0.5,
        transient_path=f'{DATA_DIR}/rendered_images/transient_images_240x320_nt-2000/'
                        f'kitchen-2_nr-240_nc-320_nt-2000_samples-2048_view-0.npz',
        depth_path=f'{DATA_DIR}/rendered_images/depth_images_240x320_nt-2000/'
                   f'kitchen-2_nr-240_nc-320_nt-2000_samples-2048_view-0.npy',
    ),
}

SCANDATA_TRES_PS = 8
SCANDATA_TLEN_PS = 10000
SCANDATA_PER_BIN_CM = SCANDATA_TRES_PS * 1e-12 * C / 2 * 100   # 0.12 cm/bin
SCANDATA_RES_T = 1144

SCANDATA_SCENES = {
    'face_scanning': SceneConfig(
        name='face_scanning', kind='scandata', R=204, Cw=116, RES_T=SCANDATA_RES_T,
        PER_BIN_CM=SCANDATA_PER_BIN_CM,
        transient_path=f'{DATA_DIR}/scan_data/20190207_face_scanning_low_mu/free/'
                        f'intermediate_results/unimodal-hist-img_r-204-c-116_tres-{SCANDATA_TRES_PS}ps_'
                        f'tlen-{SCANDATA_TLEN_PS}ps.npy',
    ),
    # Cropped to the largest multiple of ROOT=12x16 for run_adaptive.py's
    # exact-dyadic-tiling requirement: 204x116 -> 204x112.
    'face_scanning_adaptive': SceneConfig(
        name='face_scanning_adaptive', kind='scandata', R=204, Cw=112, RES_T=SCANDATA_RES_T,
        PER_BIN_CM=SCANDATA_PER_BIN_CM,
        transient_path=f'{DATA_DIR}/scan_data/20190207_face_scanning_low_mu/free/'
                        f'intermediate_results/unimodal-hist-img_r-204-c-116_tres-{SCANDATA_TRES_PS}ps_'
                        f'tlen-{SCANDATA_TLEN_PS}ps.npy',
    ),
    'deer': SceneConfig(
        name='deer', kind='scandata', R=174, Cw=154, RES_T=SCANDATA_RES_T,
        PER_BIN_CM=SCANDATA_PER_BIN_CM,
        transient_path=f'{DATA_DIR}/scan_data/20190209_deer_high_mu/free/'
                        f'intermediate_results/unimodal-hist-img_r-174-c-154_tres-{SCANDATA_TRES_PS}ps_'
                        f'tlen-{SCANDATA_TLEN_PS}ps.npy',
    ),
    # Cropped: 174x154 -> 168x144, the largest multiple of 12x16 that fits.
    'deer_adaptive': SceneConfig(
        name='deer_adaptive', kind='scandata', R=168, Cw=144, RES_T=SCANDATA_RES_T,
        PER_BIN_CM=SCANDATA_PER_BIN_CM,
        transient_path=f'{DATA_DIR}/scan_data/20190209_deer_high_mu/free/'
                        f'intermediate_results/unimodal-hist-img_r-174-c-154_tres-{SCANDATA_TRES_PS}ps_'
                        f'tlen-{SCANDATA_TLEN_PS}ps.npy',
    ),
}

ALL_SCENES = {**FLASHLIDAR_SCENES, **SCANDATA_SCENES}
