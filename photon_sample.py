"""Photon sampling: draw one photon per pixel per cycle, gate signal by p_detect.

Each scene provides five reference files, one per origin -- signal,
afterpulse, crosstalk, dark_count, background -- each a (R, Cw, RES_T)
int32 array of counts: how many photons of that origin arrived at that
pixel in that time bin, over a reference acquisition. Loading all five in
CATEGORIES order gives each pixel a (5, RES_T) table of counts, its photon
pool.

Per pixel per cycle:

    cell        = random draw over the pixel's (origin, bin) pool
    origin, bin = cell // RES_T, cell % RES_T
    if origin is signal:  detected only if gate < p_detect
    otherwise:            always detected

``sample(batch_index)`` returns the full (n_pixels, batch_size) grid of
outcomes, one entry per (pixel, cycle): the arrival bin, or -1 where that
cycle had no detection.

The generator is stateful: re-running the same configuration is
bit-identical; coding scheme, group size, shift and decoder don't touch
this file, so they can't change the photons; lowering p_detect only removes
photons (draw/gate shapes don't depend on it). Photons are NOT identical
across different B, since the generator is consumed in different-shaped
calls -- comparisons across B are statistical, not sample-for-sample.
Batches must be sampled in order from 0; a resumed run replays via skip_to.

Memory: the per-pixel CDF lives on the GPU, ~3.1 GB for a 240x320x2000
scene. Work is tiled over pixels to bound per-batch temporaries.
"""
import cupy as cp
import numpy as np

import dataset_config as dc
import load_scandata as ls

CATEGORIES = ('signal', 'afterpulse', 'crosstalk', 'dark_count', 'background')
SIGNAL_INDEX = 0

TILE_ELEMENTS = 32_000_000  # target elements per tile, to bound temporary memory


def component_path(scene_cfg, category):
    """Path of one origin's reference histogram .npz."""
    return (f'{dc.scene_sim_dir(scene_cfg)}/snr_histogram/'
            f'{category}_hist_{scene_cfg.name}.npz')


class PhotonSampler:
    """Draws the photons of one batch at a time, for a fixed scene and p_detect.

    Args:
        scene_cfg:  a dataset_config scene.
        batch_size: B, cycles per batch. B=1 is a batch of one cycle.
        p_detect:   probability in (0, 1] that a signal photon is detected.
    """

    def __init__(self, scene_cfg, batch_size, p_detect=1.0):
        if not 0.0 < p_detect <= 1.0:
            raise ValueError(f'p_detect must be in (0, 1], got {p_detect}')
        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1, got {batch_size}')
        self.scene_cfg = scene_cfg
        self.batch_size = int(batch_size)
        self.p_detect = float(p_detect)
        self.R, self.Cw, self.RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T
        self.n_pixels = self.R * self.Cw
        self.width = len(CATEGORIES) * self.RES_T

        # (n_pixels, 5, RES_T) pool on the GPU. scandata scenes have no
        # origin decomposition: the whole measured histogram goes in
        # 'signal', the other four origins stay zero.
        pool = cp.zeros((self.n_pixels, len(CATEGORIES), self.RES_T), dtype=cp.int32)
        if scene_cfg.kind == 'scandata':
            hist = ls.load_transient(scene_cfg).astype(np.int32)
            if hist.shape != (self.R, self.Cw, self.RES_T):
                raise ValueError(
                    f'signal histogram is {hist.shape}, but scene '
                    f'{scene_cfg.name} is {(self.R, self.Cw, self.RES_T)}')
            pool[:, SIGNAL_INDEX, :] = cp.asarray(hist.reshape(self.n_pixels, self.RES_T))
            del hist
        else:
            for index, category in enumerate(CATEGORIES):
                hist = np.load(component_path(scene_cfg, category))['hist']
                if hist.dtype != np.int32:
                    raise ValueError(f'{category} histogram is {hist.dtype}, expected int32')
                if hist.shape != (self.R, self.Cw, self.RES_T):
                    raise ValueError(
                        f'{category} histogram is {hist.shape}, but scene '
                        f'{scene_cfg.name} is {(self.R, self.Cw, self.RES_T)}')
                pool[:, index, :] = cp.asarray(hist.reshape(self.n_pixels, self.RES_T))
                del hist
        pool = pool.reshape(self.n_pixels, self.width)

        # Per-pixel CDF, built once. float64 cumsum for exactness over int
        # counts, then stored as float32.
        self.pool_total = pool.sum(axis=1)
        safe = cp.where(self.pool_total > 0, self.pool_total, 1).astype(cp.float64)
        self.cdf = cp.cumsum(pool.astype(cp.float64) / safe[:, None],
                             axis=1).astype(cp.float32)
        self.has_photon = self.pool_total > 0
        self.signal_share = float(
            (pool[:, :self.RES_T].sum() / pool.sum()).item())
        del pool, safe
        cp.get_default_memory_pool().free_all_blocks()

        self.rng = cp.random.default_rng(dc.EVENT_SEED)
        self.next_batch = 0
        self.accepted_by_category = np.zeros(len(CATEGORIES), dtype=np.int64)
        self.rejected = 0

    def _tile_events(self, pixel_lo, pixel_hi):
        """Arrivals for one block of pixels over the current batch's
        cycles: (pixel_hi - pixel_lo, batch_size) int32, arrival bin or -1.
        Draws from the shared generator, so must run in the same tile
        order every time.
        """
        n_rows = pixel_hi - pixel_lo
        shape = (n_rows, self.batch_size)

        # Draw a cell per pixel from its pool. searchsorted is 1-D, so
        # per-pixel CDFs are laid end to end with an offset, then the
        # offset is removed.
        draw_u = self.rng.random(shape, dtype=cp.float32)
        offset = cp.arange(n_rows, dtype=cp.float32)[:, None] * cp.float32(2.0)
        cell = cp.searchsorted(
            (self.cdf[pixel_lo:pixel_hi] + offset).ravel(),
            (draw_u + offset).ravel(), side='right')
        cell -= cp.repeat(cp.arange(n_rows, dtype=cp.int64),
                          self.batch_size) * self.width
        cell = cp.clip(cell.reshape(shape), 0, self.width - 1)
        del draw_u, offset

        origin = (cell // self.RES_T).astype(cp.int8)
        bins = (cell % self.RES_T).astype(cp.int32)
        del cell

        # Signal faces the p_detect gate; every other origin is kept.
        # gate_u is always drawn so the stream advances identically
        # regardless of p_detect.
        is_signal = origin == SIGNAL_INDEX
        gate_u = self.rng.random(shape, dtype=cp.float32)
        has_photon = self.has_photon[pixel_lo:pixel_hi][:, None]
        detected = has_photon & (~is_signal | (gate_u < self.p_detect))
        del gate_u

        self.rejected += int(cp.count_nonzero(has_photon & is_signal & ~detected))
        for value in range(len(CATEGORIES)):
            self.accepted_by_category[value] += int(
                cp.count_nonzero(detected & (origin == value)))

        return cp.where(detected, bins, cp.int32(-1))

    def _tiles(self):
        """Pixel tile boundaries, in the fixed order the generator expects."""
        rows_per_tile = max(1, TILE_ELEMENTS // self.batch_size)
        for lo in range(0, self.n_pixels, rows_per_tile):
            yield lo, min(lo + rows_per_tile, self.n_pixels)

    def sample(self, batch_index):
        """Every pixel's arrivals during one batch's cycles.

        Args:
            batch_index: must equal the next unsampled batch -- the
                generator is stateful, so batches can't be skipped or
                revisited.

        Returns:
            (n_pixels, batch_size) int32 -- arrival time bin per (pixel,
            cycle) slot, or -1 where that cycle had no detection. A pixel
            with no detections all batch is still a row of -1s.
        """
        assert batch_index == self.next_batch, (
            f'batches must be sampled in order: expected {self.next_batch}, '
            f'got {batch_index}')
        self.next_batch += 1
        tiles = [self._tile_events(lo, hi) for lo, hi in self._tiles()]
        return cp.concatenate(tiles, axis=0)

    def skip_to(self, batch_index):
        """Advance the generator to batch_index by drawing and discarding
        (the stream can't be seeked). Returns how many batches were replayed.
        """
        replayed = 0
        while self.next_batch < batch_index:
            self.sample(self.next_batch)
            replayed += 1
        self.accepted_by_category[:] = 0
        self.rejected = 0
        return replayed
