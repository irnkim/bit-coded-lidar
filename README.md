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

# CPU fallback, no GPU reachable (correctness, not speed)
BACKEND=cpu G_H=4 G_W=4 B=500 python3 run_nonadaptive.py

# K-codebook: predefine 200 codes once, each pixel each CYCLE (not batch)
# redraws a random index into them
G_H=4 G_W=4 B=500 CODE_MODE=kcodebook K_CODES=200 python3 run_nonadaptive.py

# save readout bits + brightness trace for later reuse, then also compute
# cumulative-vs-independent bilateral and iterative (1-bit CS) decode in
# the same run
G_H=4 G_W=4 B=500 CODE=trig TRIG_K=50 SAVE_BITS=1 CUM_BILATERAL=1 ICM_SWEEPS=5 \
    python3 run_nonadaptive.py

# re-decode that same measurement later -- no photon resampling, no
# recoding, just reloads the saved bits and replays the (reproducible)
# codes against them
G_H=4 G_W=4 B=500 CODE=trig TRIG_K=50 LOAD_BITS=1 CUM_BILATERAL=1 ICM_SWEEPS=5 \
    python3 run_nonadaptive.py
```

A positional argument (`python3 run_nonadaptive.py 500`) is shorthand for
`B=500`. Results land in `data/rendered_images/<scene>/results/`
as `.npz` files (depth maps, MAE, run metadata); `run_adaptive.py` also
writes a `partitions_*.npz` trace.

## Backend

`BACKEND=gpu` (default) uses cupy; every array lives in GPU VRAM.
`BACKEND=cpu` swaps in a numpy-backed shim (`xp_backend.py`) so the same
code runs with no GPU reachable at all -- every module that used to
`import cupy as cp` does `from xp_backend import cp` instead, and that's
the only thing that changes. It's a correctness fallback, not a
performance option: `codes.draw_pulse_coding_trig`'s `cos()` over a large
chunked tensor is inherently slow on single-threaded numpy (no BLAS
trick applied here, unlike GPU mode). A small smoke test at `B=10`/
`CYCLES=2000` takes under a minute on GPU and over 20 on CPU, but gives
bit-identical results. Set it before running, e.g. `BACKEND=cpu python3
run_nonadaptive.py`.

This also matters for memory: GPU mode's big per-batch tensors (the trig
code draw, the per-pixel CDF) live in VRAM, invisible to the host's own
memory limits. CPU mode puts everything in host RAM, so both of those
are computed in bounded row-chunks rather than materializing the whole
scene at once -- harmless on GPU too, just unnecessary there.

## K-codebook mode

`CODE_MODE=continuous` (default) is what every example above uses: a
fresh code per pixel, redrawn every BATCH, held fixed for its B cycles.

`CODE_MODE=kcodebook` is a different scheme: `K_CODES` distinct codes are
predefined ONCE for the whole run (shared across the scene, not
per-pixel), and every pixel, every CYCLE (not batch), independently
redraws a uniformly random index into that fixed codebook. Requires
`CODE_SHARING=unshared` (the only mode `encode.group_bits_kcodebook` /
`decode.backproject_kcodebook` support), and doesn't support
`HW_DELAY_PROBS` or `DECODE_MODE=hier0` yet.

## Reusing saved bits

`SAVE_BITS=1` writes every batch's readout bits plus a per-checkpoint
brightness image to `<result>.bits.npz`, alongside the normal result.
Codes themselves are NOT saved (a full per-batch code history would be
hundreds of GB) -- `codes.CodeSource`/`codes.KCodebookSource` are
seeded, stateful streams, so replaying one from batch 0 with the same
family/trig_k/K_CODES reproduces the exact same codes used to produce
those bits.

`LOAD_BITS=1` is the other half: point it at a config that already has a
`SAVE_BITS=1` result (same env vars, so the filename matches), and it
skips the entire acquisition loop -- no `PhotonSampler`, no new photon
draws -- loading the saved bits/brightness trace and jumping straight to
`CUM_BILATERAL`/`ICM_SWEEPS`. Use this to try a new decode against
already-measured data without resimulating.

## Alternative decoders

The production decoder is the adjoint: `decode.backproject` correlates
each group's one bit against every member pixel's own code
(`accumulator += code * bit`), then argmaxes. Two experimental
alternatives, both opt-in and both requiring `SAVE_BITS=1` or
`LOAD_BITS=1` since they replay the batch's bits rather than computing
them fresh:

- **`CUM_BILATERAL=1`** -- independent vs. cumulative (recursive)
  bilateral filtering. Independent is what `depth_maps()` already
  computes at every checkpoint: filter a COPY of the accumulator, never
  touching the stored one, so every checkpoint starts fresh from raw
  evidence. Cumulative instead REPLACES the accumulator with its
  filtered self at every checkpoint, so smoothing compounds into every
  later one. In practice cumulative looks better very early (little
  evidence yet, aggressive smoothing helps) but crosses over and then
  diverges badly once independent has enough evidence to stand on its
  own -- a textbook failure mode of recursive filtering: a checkpoint's
  smoothing error gets baked in and never corrected by later evidence.
  Writes `<result>_cumulative_vs_independent.npz` (a `(cycles,
  mae_indep, mae_cumulative, delta_pct)` trace).

- **`ICM_SWEEPS=N`** -- 1-bit compressed sensing: each pixel's true
  histogram is modeled as a single spike at an unknown bin (not a dense
  distribution), so "find `x` minimizing `||y - sign(Ax)||`" becomes a
  discrete search over that one bin per pixel. Since a group's bit mixes
  all its member pixels, it's solved by Jacobi-style iterative
  coordinate descent: freeze every pixel's current bin estimate, then
  for every pixel in parallel try all `RES_T` candidate bins and keep
  whichever reproduces the most observed batch signs (holding the other
  group members at their frozen estimates), for N sweeps. Reports three
  variants per sweep, writing `<result>_iterative_decode.npz`:
  - `ICM-recon` -- raw argmax of each sweep's match-count evidence, fed
    forward unfiltered into the next sweep.
  - `ICM-bilat` -- an independent bilateral filter of that same
    evidence, every sweep, never fed back.
  - `ICM-cum-bilat` -- a cumulative bilateral filter that IS fed
    forward into the next sweep's coordinate descent (the same
    independent-vs-cumulative distinction as `CUM_BILATERAL`, applied to
    ICM's own evidence instead of the linear accumulator).

  `CODE_MODE=kcodebook` isn't supported here yet -- each cycle's code
  differs within a batch in that mode, which this scoring doesn't yet
  account for.

## Configuration

Shared (both drivers):

| Env var | Values | Default |
|---|---|---|
| `SCENE` | any `dataset_config.ALL_SCENES` key | `bathroom-cycles-2` |
| `B` | cycles per batch | `1` |
| `CODE` | `bernoulli`, `lowpass`, `trig`, `chipoffset` -- also accepts `filtered-bernoulli`/`cosine-wave`/`wide-bernoulli` as aliases for `lowpass`/`trig`/`chipoffset` (see `codes.FAMILY_ALIASES`); all three non-bernoulli families are themselves bernoulli-based (smoothed, summed-cosine, or chip-grouped), resolved to the canonical name before touching any filename, so results/checkpoints are unaffected either way | `bernoulli` |
| `CODE_SHARING` | `unshared`, `shared` | `unshared` |
| `P_DETECT` | signal detection probability, `(0, 1]` | `1.0` |
| `CYCLES` | total illumination cycles | `dataset_config.M` (500,000) |
| `HW_DELAY_PROBS` | `"0:0.7,1:0.2,2:0.1"` (delay-in-bins:fraction, sums to 1) | off |
| `HW_DELAY_MODE` | `per_cycle` (independent delay draw per pixel/cycle) or `per_batch_group` (one draw per batch/group, shared by every pixel in the group and every cycle in the batch) | `per_cycle` |
| `TRIG_K` | harmonic count (frequency, F) for `CODE=trig`/`cosine-wave` only, ignored by every other family | `3` |
| `BILAT_RADIUS` | bilateral filter window half-width (px) | `5` |
| `BILAT_SIGMA_SPATIAL` | bilateral spatial falloff | `2.0` |
| `BILAT_SIGMA_RANGE` | bilateral brightness falloff | `0.15` |
| `VERIFY_BATCHES` | stop after N batches, skip ground truth (smoke test) | off |
| `CHECKPOINT_EVERY` | batches between checkpoints | `500` |
| `OUT_TAG` | suffix appended to the output filename | `''` |
| `BACKEND` | `gpu` (cupy) or `cpu` (numpy fallback -- see [Backend](#backend)) | `gpu` |

`run_nonadaptive.py` only:

| Env var | Values | Default |
|---|---|---|
| `G_H`, `G_W` | group height/width in pixels | `1`, `1` |
| `SHIFT_STEP` | `0` fixed block, `1` sliding block | `0` |
| `DECODE_MODE` | `majority0`, `hier0` | `majority0` |
| `CODE_MODE` | `continuous` (default: a fresh code per BATCH, held fixed for its B cycles) or `kcodebook` (see [K-codebook mode](#k-codebook-mode)) | `continuous` |
| `K_CODES` | codebook size; required (`> 0`) when `CODE_MODE=kcodebook`, ignored otherwise | `0` |
| `SAVE_BITS` | `1` to also save every batch's readout bits + a per-checkpoint brightness image to `<result>.bits.npz` (see [Reusing saved bits](#reusing-saved-bits)); requires `SHIFT_STEP=0` | off |
| `LOAD_BITS` | `1` to **skip the acquisition loop entirely** and load a prior `SAVE_BITS=1` run's bits instead -- for `CUM_BILATERAL`/`ICM_SWEEPS` against already-measured data without redrawing photons or codes | off |
| `CUM_BILATERAL` | `1` to also compute a cumulative (recursive) vs. independent bilateral-filter MAE trace (see [Alternative decoders](#alternative-decoders)); requires `SAVE_BITS=1` or `LOAD_BITS=1` | off |
| `ICM_SWEEPS` | `> 0` to also run that many iterative-decode (1-bit compressed sensing) sweeps (see [Alternative decoders](#alternative-decoders)); requires `SAVE_BITS=1` or `LOAD_BITS=1`, and `CODE_MODE=continuous` | `0` |

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
