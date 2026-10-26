"""Online spatial hierarchy driven only by legal one-bit group measurements.

The acquisition loop supplies cumulative 3x4-cell correlation sketches. Peaks
are estimated from the normalized union A+B; the odd/even A and B histories
decide only whether that estimate is repeatable. No RGB image, ground-truth
depth, per-cycle readout, or hidden sub-group measurement enters this module.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


MIN_H, MIN_W = 3, 4


@dataclass
class CellEvidence:
    primary: np.ndarray
    primary_a: np.ndarray
    primary_b: np.ndarray
    reliable: np.ndarray
    secondary: np.ndarray
    secondary_ratio: np.ndarray


@dataclass
class NodeDecision:
    state: str
    n_reliable: int
    n_required: int
    n_plane: int
    k_plane: int
    n_secondary: int
    max_secondary: float

    @property
    def structured(self):
        return self.state == 'structured'


def _normalized_smooth(raw, count, sigma):
    if count <= 0:
        return np.zeros_like(raw, dtype=np.float32)
    value = np.asarray(raw, dtype=np.float32) / float(count)
    return gaussian_filter1d(value, sigma, axis=-1, mode='nearest')


def extract_cell_evidence(raw_a, raw_b, n_a, n_b, sigma=2.0,
                          agreement_bins=8, exclusion_bins=6):
    """Extract peaks once per 3x4 candidate cell.

    ``raw_a`` and ``raw_b`` are cumulative correlation sums, not separate
    reconstructions.  The combined profile is exactly

        (raw_a + raw_b) / (n_a + n_b),

    so unequal odd/even counts are handled correctly.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError('both cumulative histories need at least one batch')
    raw_a = np.asarray(raw_a, dtype=np.float32)
    raw_b = np.asarray(raw_b, dtype=np.float32)
    if raw_a.shape != raw_b.shape or raw_a.ndim != 3:
        raise ValueError('cell sketches must have matching (rows, cols, time) shapes')

    smooth_a = _normalized_smooth(raw_a, n_a, sigma)
    smooth_b = _normalized_smooth(raw_b, n_b, sigma)
    smooth = gaussian_filter1d(
        (raw_a + raw_b) / float(n_a + n_b), sigma, axis=-1, mode='nearest'
    )
    primary_a = np.argmax(smooth_a, axis=-1).astype(np.int16)
    primary_b = np.argmax(smooth_b, axis=-1).astype(np.int16)
    primary = np.argmax(smooth, axis=-1).astype(np.int16)
    reliable = np.abs(primary_a.astype(np.int32) - primary_b.astype(np.int32)) <= agreement_bins

    # Find the strongest baseline-subtracted return outside the primary's
    # exclusion window.  It need not be a "thin" object: it means only that
    # two depth populations contributed to this 3x4 candidate cell.
    baseline = np.median(smooth, axis=-1, keepdims=True)
    signal = np.maximum(smooth - baseline, 0.0)
    time = np.arange(smooth.shape[-1], dtype=np.int32)
    outside = np.abs(time[None, None, :] - primary[..., None]) >= exclusion_bins
    secondary_signal = np.where(outside, signal, -np.inf)
    secondary = np.argmax(secondary_signal, axis=-1).astype(np.int16)
    primary_height = np.take_along_axis(signal, primary[..., None], axis=-1)[..., 0]
    secondary_height = np.take_along_axis(signal, secondary[..., None], axis=-1)[..., 0]
    ratio = np.divide(secondary_height, primary_height,
                      out=np.zeros_like(secondary_height), where=primary_height > 0)
    return CellEvidence(primary, primary_a, primary_b, reliable, secondary, ratio)


def decide_node(evidence, row0, col0, cells_h, cells_w,
                plane_tolerance=4, secondary_ratio=0.20,
                reliable_fraction=0.25):
    """Return a plane-only structured/smooth/uncertain decision.

    Secondary-peak statistics remain in the returned diagnostics, but they do
    not affect the partition state.
    """
    sl = np.s_[row0:row0 + cells_h, col0:col0 + cells_w]
    peaks = evidence.primary[sl].astype(np.float64)
    reliable = evidence.reliable[sl]
    ratios = evidence.secondary_ratio[sl]
    n_cells = int(cells_h * cells_w)
    n_reliable = int(np.count_nonzero(reliable))
    n_required = max(2, int(np.ceil(reliable_fraction * n_cells)))
    k_plane = max(2, int(np.ceil(0.05 * n_cells)))
    if n_reliable < n_required:
        return NodeDecision('uncertain', n_reliable, n_required, 0, k_plane, 0, 0.0)

    rr, cc = np.meshgrid(np.linspace(-1.0, 1.0, cells_h),
                         np.linspace(-1.0, 1.0, cells_w), indexing='ij')
    design = np.stack((np.ones_like(rr), rr, cc), axis=-1)
    theta, *_ = np.linalg.lstsq(design[reliable], peaks[reliable], rcond=None)
    residual = np.abs(peaks - design @ theta)
    n_plane = int(np.count_nonzero(reliable & (residual > plane_tolerance)))
    supported_secondary = reliable & (ratios > secondary_ratio)
    n_secondary = int(np.count_nonzero(supported_secondary))
    max_secondary = float(np.max(np.where(reliable, ratios, 0.0)))
    structured = n_plane >= k_plane
    return NodeDecision('structured' if structured else 'smooth',
                        n_reliable, n_required, n_plane, k_plane,
                        n_secondary, max_secondary)


class OnlineHierarchy:
    """24x32 -> 12x16 -> 6x8 -> 3x4 split/merge state machine."""

    def __init__(self, image_h, image_w, root_h=24, root_w=32,
                 split_confirmations=2, merge_confirmations=3,
                 initially_split=True):
        if (root_h, root_w) not in {(12, 16), (24, 32)}:
            raise ValueError('root must be 12x16 or 24x32')
        if image_h % root_h or image_w % root_w:
            raise ValueError('root must tile image exactly')
        self.image_h, self.image_w = image_h, image_w
        self.root_h, self.root_w = root_h, root_w
        self.split_confirmations = split_confirmations
        self.merge_confirmations = merge_confirmations
        self.levels = [(root_h, root_w)]
        if (root_h, root_w) == (24, 32):
            self.levels.append((12, 16))
        self.levels.append((6, 8))
        self.split = {}
        self.structured_streak = {}
        self.smooth_streak = {}
        for h, w in self.levels:
            shape = (image_h // h, image_w // w)
            self.split[(h, w)] = np.full(shape, initially_split, dtype=bool)
            self.structured_streak[(h, w)] = np.zeros(shape, dtype=np.int16)
            self.smooth_streak[(h, w)] = np.zeros(shape, dtype=np.int16)

    def _observe(self, level, index, decision):
        if decision.state == 'uncertain':
            self.structured_streak[level][index] = 0
            self.smooth_streak[level][index] = 0
            return
        if decision.state == 'structured':
            self.structured_streak[level][index] += 1
            self.smooth_streak[level][index] = 0
            if self.structured_streak[level][index] >= self.split_confirmations:
                self.split[level][index] = True
        else:
            self.smooth_streak[level][index] += 1
            self.structured_streak[level][index] = 0
            if self.smooth_streak[level][index] >= self.merge_confirmations:
                self.split[level][index] = False

    def update(self, evidence):
        decisions = {}
        # Visibility is evaluated from the partition at the start of this update.
        # A newly split parent therefore exposes its children only at the next
        # update, preventing a coarse-start tree from jumping several levels at once.
        visible = {}
        for h, w in self.levels:
            level = (h, w)
            if level == self.levels[0]:
                visible[level] = np.ones_like(self.split[level], dtype=bool)
            else:
                parent = (2 * h, 2 * w)
                parent_visible = visible[parent]
                parent_open = parent_visible & self.split[parent]
                visible[level] = np.repeat(
                    np.repeat(parent_open, 2, axis=0), 2, axis=1
                )
            ch, cw = h // MIN_H, w // MIN_W
            for i in range(self.image_h // h):
                for j in range(self.image_w // w):
                    if not visible[level][i, j]:
                        self.structured_streak[level][i, j] = 0
                        self.smooth_streak[level][i, j] = 0
                        continue
                    d = decide_node(evidence, i * ch, j * cw, ch, cw)
                    decisions[(h, w, i, j)] = d
                    self._observe(level, (i, j), d)
        return decisions

    def leaves(self):
        leaves = []

        def descend(r, c, h, w):
            if (h, w) == (MIN_H, MIN_W):
                leaves.append((r, c, h, w))
                return
            level = (h, w)
            i, j = r // h, c // w
            if not self.split[level][i, j]:
                leaves.append((r, c, h, w))
                return
            nh, nw = h // 2, w // 2
            for dr in (0, nh):
                for dc in (0, nw):
                    descend(r + dr, c + dc, nh, nw)

        for r in range(0, self.image_h, self.root_h):
            for c in range(0, self.image_w, self.root_w):
                descend(r, c, self.root_h, self.root_w)
        return leaves

    def state_dict(self):
        result = {}
        for h, w in self.levels:
            tag = f'{h}x{w}'
            result[f'hier_split_{tag}'] = self.split[(h, w)]
            result[f'hier_structured_streak_{tag}'] = self.structured_streak[(h, w)]
            result[f'hier_smooth_streak_{tag}'] = self.smooth_streak[(h, w)]
        return result

    def load_state_dict(self, state):
        for h, w in self.levels:
            tag = f'{h}x{w}'
            for prefix, destination in (
                    ('hier_split_', self.split),
                    ('hier_structured_streak_', self.structured_streak),
                    ('hier_smooth_streak_', self.smooth_streak)):
                key = prefix + tag
                if key in state:
                    destination[(h, w)][...] = np.asarray(state[key])


def leaf_counts(leaves):
    counts = {(24, 32): 0, (12, 16): 0, (6, 8): 0, (3, 4): 0}
    for _, _, h, w in leaves:
        counts[(h, w)] = counts.get((h, w), 0) + 1
    return counts
