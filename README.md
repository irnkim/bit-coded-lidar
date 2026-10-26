# Bit-Coded LiDAR

Simulated acquisition for a coded single-photon LiDAR: pixels are grouped
into blocks, each block shares one physical detector, and the sensor emits
one ternary (majority-vote) bit per group per batch. Two drivers cover every
configuration:

- `run_nonadaptive.py` -- fixed group size (G) and batch size (B) for the
  whole run. G=1x1 is the transient/temporal case; G>1x1 is spatial pooling.
- `run_adaptive.py` -- same acquisition, plus a controller that grows/shrinks
  the group partition online from decoded evidence (no ground truth).

## Install

```
pip install -r requirements.txt
```

Requires a CUDA GPU (cupy-cuda12x). Scene data is not in this repo (too
large for git) -- download `bathroom-cycles-2` from
[Google Drive](https://drive.google.com/drive/folders/1QpCOi8PI_fjXvehxPy0F4tPu0ivpRtn1?usp=drive_link)
and extract it to `data/`, matching the layout in [Data paths](#data-paths).

## Data paths

All paths are set in `dataset_config.py`, relative to the repo by default
(`DATA_DIR`/`SIM_DIR` = `<repo>/data`, `RESULTS_DIR` = `<repo>/results`). To
point at a different location, edit these. Layout differs by scene `kind`:

- **`DATA_DIR` / `SIM_DIR`** (top of `dataset_config.py`) -- the input root.
  Change this to move where scene data is read from.
- **`kind='flashlidar'`** (`bathroom-cycles-2`, `kitchen-2`) -- a rendered
  scene, split into ground truth and a separately-labelled noise histogram:
  - **Ground truth**: `{SIM_DIR}/rendered_images/<scene>/depth_gt.npz`, from
    `scene_sim_dir()` and read by `reconstruct.ground_truth_path()`.
  - **SNR histogram** (the five per-origin reference files `photon_sample.py`
    loads): `{SIM_DIR}/rendered_images/<scene>/snr_histogram/<category>_hist_
    <scene>.npz`, from `photon_sample.component_path()`.
- **`kind='scandata'`** (`face_scanning`, `deer`, and their `_adaptive`
  crops) -- a real measured histogram, one file, no separate GT file (ground
  truth is the argmax of that same histogram, via
  `load_scandata.load_gt_depth_cm`): set `transient_path` directly on the
  scene's `SceneConfig` entry in `dataset_config.py`, under `DATA_DIR`.
- **Saving results** (either kind): `{SIM_DIR}/<rendered_images or
  scan_data>/<scene>/results/`, built in `output_path()`
  (`run_nonadaptive.py`) / `output_paths()` (`run_adaptive.py`). Change
  `SIM_DIR` (or `scene_sim_dir()` itself) to save elsewhere.

Adding a new scene means adding a `SceneConfig` entry to `FLASHLIDAR_SCENES`
or `SCANDATA_SCENES` in `dataset_config.py` and placing its data at the
paths above for that kind.

## Run

```
# transient baseline (no pooling)
G_H=1 G_W=1 B=1 python3 run_nonadaptive.py

# temporal pooling only
G_H=1 G_W=1 B=100 python3 run_nonadaptive.py

# spatial pooling, fixed 3x4 block
G_H=3 G_W=4 B=1 python3 run_nonadaptive.py

# spatial + temporal pooling
G_H=3 G_W=4 B=500 python3 run_nonadaptive.py

# jitter-robust code family, reduced detection probability
G_H=3 G_W=4 B=500 CODE=lowpass P_DETECT=0.3 python3 run_nonadaptive.py

# hardware code-distribution delay
G_H=3 G_W=4 B=500 HW_DELAY_PROBS="0:0.25,1:0.25,2:0.25,3:0.25" python3 run_nonadaptive.py

# adaptive partition, scale-pure with median filter
B=100 SCALE=pure PIXEL_MEDIAN_SIZE=3 python3 run_adaptive.py
```

A positional argument (`python3 run_nonadaptive.py 500`) is shorthand for
`B=500`. Results land in `data/rendered_images/<scene>/results/`
as `.npz` files (depth maps, MAE, run metadata); `run_adaptive.py` also
writes a `partitions_*.npz` trace.

## Configuration

Shared (both drivers):

| Env var | Values | Default |
|---|---|---|
| `SCENE` | any `dataset_config.ALL_SCENES` key | `bathroom-cycles-2` |
| `B` | cycles per batch | `1` |
| `CODE` | `bernoulli`, `lowpass`, `trig`, `chipoffset` | `bernoulli` |
| `CODE_SHARING` | `unshared`, `shared` | `unshared` |
| `P_DETECT` | signal detection probability, `(0, 1]` | `1.0` |
| `CYCLES` | total illumination cycles | `dataset_config.M` (500,000) |
| `HW_DELAY_PROBS` | `"0:0.7,1:0.2,2:0.1"` (delay-in-bins:fraction, sums to 1) | off |
| `BILAT_RADIUS` | bilateral filter window half-width (px) | `5` |
| `BILAT_SIGMA_SPATIAL` | bilateral spatial falloff | `2.0` |
| `BILAT_SIGMA_RANGE` | bilateral brightness falloff | `0.15` |
| `VERIFY_BATCHES` | stop after N batches, skip ground truth (smoke test) | off |
| `CHECKPOINT_EVERY` | batches between checkpoints | `500` |
| `OUT_TAG` | suffix appended to the output filename | `''` |

`run_nonadaptive.py` only:

| Env var | Values | Default |
|---|---|---|
| `G_H`, `G_W` | group height/width in pixels | `1`, `1` |
| `SHIFT_STEP` | `0` fixed block, `1` sliding block | `0` |
| `DECODE_MODE` | `majority0`, `hier0` | `majority0` |

`run_adaptive.py` only:

| Env var | Values | Default |
|---|---|---|
| `ROOT` | starting block, `24x32` or `12x16` | `24x32` |
| `START` | `finest`, `coarsest` | `finest` |
| `UPDATE_EVERY` | batches between controller updates | `10` |
| `AUDIT_BATCHES` | forced full-resolution audit batches per window (even, `< UPDATE_EVERY`) | `2` |
| `SCALE` | `pure` (finest-scale pixels only), `all` | `pure` |
| `PIXEL_MIN_COUNT` | min accepted measurements per half before trusting a pixel | `20` |
| `PIXEL_SUPPORT` | min per-half peak z-score | `1.5` |
| `PIXEL_MEDIAN_SIZE` | median filter on decoded depth (`1` disables) | `3` |
| `PIXEL_RESIDUAL_BINS` | plane-residual outlier threshold, in bins | `4` |
| `PIXEL_SPLIT_FRAC` | outlier fraction that signals split | `0.05` |
| `PIXEL_MERGE_FRAC` | outlier fraction that signals merge | `0.02` |
| `PIXEL_MIN_FRAC` | reliable-pixel fraction needed to judge a block | `0.50` |
| `SPLIT_CONFIRM` | consecutive split signals required | `2` |
| `MERGE_CONFIRM` | consecutive merge signals required | `3` |
