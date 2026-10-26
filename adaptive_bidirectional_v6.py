"""Bidirectional, measurement-only adaptive spatial hierarchy (v13).

Every acquisition uses a legal spatial partition: one majority/sign bit is
returned for each leaf.  The normal layout is the current partition.  The
refinement-audit layout replaces every non-minimum leaf by its four children.
Consequently a merged node continues to receive direct-child evidence capable
of reversing the merge, while a split node receives direct-child measurements
from its normal layout.

Evidence is kept separately for the group size that was actually measured.
Every pixel uses a fresh independent temporal code; the one legal group bit is
matched-filtered by those known per-pixel codes and then averaged digitally.
For a parent test, these legal unshared-code backprojections are averaged within
each of its four directly measured children.  A held-out window compares the
four child-peak predictions with a smooth planar-parent prediction.  Its
predictive gain is accumulated in a signed score, then a penalized
dynamic-programming tree cut chooses the leaves.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from adaptive_online_onebit import decide_node, extract_cell_evidence


MIN_H, MIN_W = 3, 4


@dataclass
class ScaleEvidence:
    primary: np.ndarray
    primary_a: np.ndarray
    primary_b: np.ndarray
    reliable: np.ndarray
    support_a: np.ndarray
    support_b: np.ndarray
    smooth_a: np.ndarray
    smooth_b: np.ndarray
    count_a: np.ndarray
    count_b: np.ndarray
    min_count: int
    support_threshold: float


@dataclass
class NodeUpdate:
    valid: bool
    n_reliable: int
    n_required: int
    predictive_gain: float
    delta: float
    score_before: float
    score_after: float
    state_before: str
    state_after: str
    transition: str


def _normalized_smooth(raw, count, sigma):
    raw = np.asarray(raw, dtype=np.float32)
    count = np.asarray(count, dtype=np.float32)
    if raw.ndim != 3 or count.shape != raw.shape[:2]:
        raise ValueError('history must be (rows, cols, time) with matching counts')
    normalized = np.divide(
        raw,
        count[..., None],
        out=np.zeros_like(raw, dtype=np.float32),
        where=count[..., None] > 0,
    )
    return gaussian_filter1d(normalized, sigma, axis=-1, mode='nearest')


def _profile_utility_array(profiles, bins):
    """Robust z-score at one selected bin per profile."""
    center = np.median(profiles, axis=-1)
    scale = 1.4826 * np.median(
        np.abs(profiles - center[..., None]), axis=-1
    )
    fallback = np.std(profiles, axis=-1)
    scale = np.where(scale > 1e-6, scale, fallback)
    selected = np.take_along_axis(profiles, bins[..., None], axis=-1)[..., 0]
    return np.divide(
        selected - center, scale,
        out=np.zeros_like(selected, dtype=np.float32), where=scale > 1e-6,
    )


def extract_scale_evidence(history, sigma=2.0, min_count=20,
                           support_threshold=1.5):
    """Decode A+B cell peaks from bits acquired at exactly one group scale.

    A and B do not independently select competing peaks.  They validate the
    peak selected by their normalized sum by both supporting that same bin.
    This avoids a 2,000-way argmax agreement test dominated by noise maxima.
    """
    raw_a = np.asarray(history['a'], dtype=np.float32)
    raw_b = np.asarray(history['b'], dtype=np.float32)
    count_a = np.asarray(history['na'], dtype=np.int32)
    count_b = np.asarray(history['nb'], dtype=np.int32)
    if raw_a.shape != raw_b.shape:
        raise ValueError('A/B histories must have identical shapes')
    smooth_a = _normalized_smooth(raw_a, count_a, sigma)
    smooth_b = _normalized_smooth(raw_b, count_b, sigma)
    total = count_a + count_b
    combined = gaussian_filter1d(
        np.divide(
            raw_a + raw_b,
            total[..., None],
            out=np.zeros_like(raw_a, dtype=np.float32),
            where=total[..., None] > 0,
        ),
        sigma,
        axis=-1,
        mode='nearest',
    )
    primary_a = np.argmax(smooth_a, axis=-1).astype(np.int16)
    primary_b = np.argmax(smooth_b, axis=-1).astype(np.int16)
    primary = np.argmax(combined, axis=-1).astype(np.int16)
    support_a = _profile_utility_array(smooth_a, primary)
    support_b = _profile_utility_array(smooth_b, primary)
    reliable = (
        (count_a >= int(min_count))
        & (count_b >= int(min_count))
        & (support_a >= float(support_threshold))
        & (support_b >= float(support_threshold))
    )
    return ScaleEvidence(
        primary, primary_a, primary_b, reliable, support_a, support_b,
        smooth_a, smooth_b, count_a, count_b, int(min_count),
        float(support_threshold),
    )


def prepare_pyramid(histories, sigma=2.0, min_count=20,
                    support_threshold=1.5):
    return {
        level: extract_scale_evidence(
            history, sigma, min_count, support_threshold
        )
        for level, history in histories.items()
    }


def _profile_utility(profile, time_bin):
    """Robust dimensionless support for one predicted time bin."""
    center = float(np.median(profile))
    scale = float(np.median(np.abs(profile - center))) * 1.4826
    if scale < 1e-6:
        scale = float(np.std(profile))
    if scale < 1e-6:
        return 0.0
    index = int(np.clip(round(float(time_bin)), 0, profile.shape[-1] - 1))
    return (float(profile[index]) - center) / scale


def predictive_plane_gain(train, window, row, col, height, width,
                          source_level,
                          complexity_penalty=0.0,
                          reliable_fraction=1.0):
    """Held-out four-child gain over a planar parent model.

    The direct-child acquisition scale supplies legal matched-filter histories
    from independent per-pixel codes.  Average their 3x4 sketches digitally
    within each of the parent's four measured children.  New-window A/B child
    profiles score the accumulated child peaks against an affine plane.  Only
    bits actually acquired at the direct-child scale enter the test.
    """
    cell_h, cell_w = 3, 4
    source_h, source_w = source_level
    if (source_h, source_w) != (height // 2, width // 2):
        raise ValueError('plane test requires the direct-child measurement scale')
    groups_h, groups_w = height // source_h, width // source_w
    train_a = np.zeros((groups_h, groups_w, train.smooth_a.shape[-1]), np.float32)
    train_b = np.zeros_like(train_a)
    window_a = np.zeros_like(train_a)
    window_b = np.zeros_like(train_a)
    train_count_a = np.zeros((groups_h, groups_w), np.int32)
    train_count_b = np.zeros_like(train_count_a)
    window_count_a = np.zeros_like(train_count_a)
    window_count_b = np.zeros_like(train_count_a)
    for group_row in range(groups_h):
        for group_col in range(groups_w):
            r0 = (row + group_row * source_h) // cell_h
            c0 = (col + group_col * source_w) // cell_w
            r1 = r0 + source_h // cell_h
            c1 = c0 + source_w // cell_w
            sl = np.s_[r0:r1, c0:c1]
            train_a[group_row, group_col] = train.smooth_a[sl].mean(axis=(0, 1))
            train_b[group_row, group_col] = train.smooth_b[sl].mean(axis=(0, 1))
            window_a[group_row, group_col] = window.smooth_a[sl].mean(axis=(0, 1))
            window_b[group_row, group_col] = window.smooth_b[sl].mean(axis=(0, 1))
            # Every 3x4 sketch inside one aligned measured child is updated by
            # the same legal group readout, hence its scale-specific counts are
            # identical.  Min is defensive against incomplete saved states.
            train_count_a[group_row, group_col] = int(train.count_a[sl].min())
            train_count_b[group_row, group_col] = int(train.count_b[sl].min())
            window_count_a[group_row, group_col] = int(window.count_a[sl].min())
            window_count_b[group_row, group_col] = int(window.count_b[sl].min())

    total = train_count_a + train_count_b
    combined = np.divide(
        train_a * train_count_a[..., None] + train_b * train_count_b[..., None],
        total[..., None], out=np.zeros_like(train_a), where=total[..., None] > 0,
    )
    peaks = np.argmax(combined, axis=-1).astype(np.int16)
    support_a = _profile_utility_array(train_a, peaks)
    support_b = _profile_utility_array(train_b, peaks)
    reliable = (
        (train_count_a >= train.min_count)
        & (train_count_b >= train.min_count)
        & (support_a >= train.support_threshold)
        & (support_b >= train.support_threshold)
    )
    n_reliable = int(np.count_nonzero(reliable))
    n_required = max(4, int(np.ceil(reliable_fraction * groups_h * groups_w)))
    window_available = (window_count_a + window_count_b) > 0
    evaluated = reliable & window_available
    valid = n_reliable >= n_required and np.count_nonzero(evaluated) >= n_required
    if not valid:
        return False, n_reliable, n_required, 0.0, 0.0

    peaks = peaks.astype(np.float64)
    rr, cc = np.meshgrid(
        np.linspace(-1.0, 1.0, groups_h),
        np.linspace(-1.0, 1.0, groups_w), indexing='ij'
    )
    design = np.stack((np.ones_like(rr), rr, cc), axis=-1)
    theta, *_ = np.linalg.lstsq(design[reliable], peaks[reliable], rcond=None)
    plane = np.clip(design @ theta, 0, train.smooth_a.shape[-1] - 1)

    child_score = 0.0
    plane_score = 0.0
    n_profiles = 0
    for local_row in range(groups_h):
        for local_col in range(groups_w):
            if not evaluated[local_row, local_col]:
                continue
            child_bin = peaks[local_row, local_col]
            plane_bin = plane[local_row, local_col]
            for profile, count in (
                    (window_a[local_row, local_col],
                     window_count_a[local_row, local_col]),
                    (window_b[local_row, local_col],
                     window_count_b[local_row, local_col])):
                if count <= 0:
                    continue
                child_score += _profile_utility(profile, child_bin)
                plane_score += _profile_utility(profile, plane_bin)
                n_profiles += 1
    if n_profiles == 0:
        return False, n_reliable, n_required, 0.0, 0.0
    predictive_gain = (child_score - plane_score) / float(n_profiles)
    delta = predictive_gain - float(complexity_penalty)
    return True, n_reliable, n_required, predictive_gain, delta


def direct_child_plane_residual(evidence, row, col, height, width,
                                source_level):
    """Normalized four-child plane residual from cumulative legal histories.

    The peak of each direct child is selected from normalized A+B.  A and B
    validate that selected peak; they do not independently choose the depth.
    The residual is normalized by A/B peak disagreement so a noisy four-peak
    saddle is not treated like reliable scene structure.
    """
    source_h, source_w = source_level
    if (source_h, source_w) != (height // 2, width // 2):
        raise ValueError('residual test requires the direct-child scale')
    profiles_a, profiles_b = [], []
    counts_a, counts_b = [], []
    for dr in (0, source_h):
        for dc in (0, source_w):
            row0 = (row + dr) // MIN_H
            col0 = (col + dc) // MIN_W
            row1 = row0 + source_h // MIN_H
            col1 = col0 + source_w // MIN_W
            sl = np.s_[row0:row1, col0:col1]
            profiles_a.append(evidence.smooth_a[sl].mean(axis=(0, 1)))
            profiles_b.append(evidence.smooth_b[sl].mean(axis=(0, 1)))
            counts_a.append(int(evidence.count_a[sl].min()))
            counts_b.append(int(evidence.count_b[sl].min()))
    profiles_a = np.asarray(profiles_a)
    profiles_b = np.asarray(profiles_b)
    counts_a = np.asarray(counts_a)
    counts_b = np.asarray(counts_b)
    total = counts_a + counts_b
    combined = np.divide(
        profiles_a * counts_a[:, None] + profiles_b * counts_b[:, None],
        total[:, None], out=np.zeros_like(profiles_a),
        where=total[:, None] > 0,
    )
    peaks = np.argmax(combined, axis=-1).astype(np.float64)
    support_a = _profile_utility_array(profiles_a, peaks.astype(np.int16))
    support_b = _profile_utility_array(profiles_b, peaks.astype(np.int16))
    reliable = (
        (counts_a >= evidence.min_count)
        & (counts_b >= evidence.min_count)
        & (support_a >= evidence.support_threshold)
        & (support_b >= evidence.support_threshold)
    )
    n_reliable = int(np.count_nonzero(reliable))
    if n_reliable < 4:
        return False, n_reliable, 4, 0.0

    design = np.asarray([
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ])
    theta, *_ = np.linalg.lstsq(design, peaks, rcond=None)
    residual = float(np.mean(np.abs(peaks - design @ theta)))
    peaks_a = np.argmax(profiles_a, axis=-1).astype(np.float64)
    peaks_b = np.argmax(profiles_b, axis=-1).astype(np.float64)
    uncertainty = max(float(np.median(np.abs(peaks_a - peaks_b))) / 2.0, 1.0)
    return True, n_reliable, 4, residual / uncertainty


class BidirectionalHierarchy:
    """24x32 -> 12x16 -> 6x8 -> 3x4 online tree controller."""

    def __init__(self, image_h, image_w, root_h=24, root_w=32,
                 initially_split=True, rho=0.8,
                 split_threshold=0.1, merge_threshold=0.1,
                 complexity_penalty=0.0, leaf_penalty=0.03,
                 min_train_count=5, count_scale_exponent=0.0,
                 support_threshold=1.5,
                 score_cap=20.0):
        if (root_h, root_w) not in {(12, 16), (24, 32)}:
            raise ValueError('root must be 12x16 or 24x32')
        if image_h % root_h or image_w % root_w:
            raise ValueError('root must tile image exactly')
        if not 0 <= rho <= 1:
            raise ValueError('rho must lie in [0, 1]')
        self.image_h, self.image_w = image_h, image_w
        self.root_h, self.root_w = root_h, root_w
        self.rho = float(rho)
        self.split_threshold = float(split_threshold)
        self.merge_threshold = float(merge_threshold)
        self.complexity_penalty = float(complexity_penalty)
        self.leaf_penalty = float(leaf_penalty)
        self.min_train_count = int(min_train_count)
        self.count_scale_exponent = float(count_scale_exponent)
        self.support_threshold = float(support_threshold)
        self.score_cap = float(score_cap)
        self.levels = [(root_h, root_w)]
        if (root_h, root_w) == (24, 32):
            self.levels.append((12, 16))
        self.levels.append((6, 8))
        self.group_levels = self.levels + [(MIN_H, MIN_W)]
        self.split = {}
        self.score = {}
        self.evidence_updates = {}
        for h, w in self.levels:
            shape = (image_h // h, image_w // w)
            self.split[(h, w)] = np.full(shape, initially_split, dtype=bool)
            self.score[(h, w)] = np.zeros(shape, dtype=np.float32)
            self.evidence_updates[(h, w)] = np.zeros(shape, dtype=np.int32)

    def _is_leaf(self, row, col, height, width):
        if (height, width) == (MIN_H, MIN_W):
            return True
        return not bool(self.split[(height, width)][row // height, col // width])

    def leaves(self):
        result = []

        def descend(row, col, height, width):
            if self._is_leaf(row, col, height, width):
                result.append((row, col, height, width))
                return
            child_h, child_w = height // 2, width // 2
            for dr in (0, child_h):
                for dc in (0, child_w):
                    descend(row + dr, col + dc, child_h, child_w)

        for row in range(0, self.image_h, self.root_h):
            for col in range(0, self.image_w, self.root_w):
                descend(row, col, self.root_h, self.root_w)
        return result

    def _children_are_leaves(self, row, col, height, width):
        child_h, child_w = height // 2, width // 2
        return all(
            self._is_leaf(row + dr, col + dc, child_h, child_w)
            for dr in (0, child_h) for dc in (0, child_w)
        )

    def decision_nodes(self):
        """Return the union of exposed split and merge candidates.

        A split parent with four leaf children is a merge candidate, while
        each non-minimum child is simultaneously a possible split candidate.
        This overlap is intentional: positive child evidence blocks the parent
        merge during transition resolution.
        """
        result = set()

        def descend(row, col, height, width):
            if (height, width) == (MIN_H, MIN_W):
                return
            level = (height, width)
            index = (row // height, col // width)
            if not self.split[level][index]:
                result.add((row, col, height, width))
                return
            if self._children_are_leaves(row, col, height, width):
                result.add((row, col, height, width))
            child_h, child_w = height // 2, width // 2
            for dr in (0, child_h):
                for dc in (0, child_w):
                    descend(row + dr, col + dc, child_h, child_w)

        for row in range(0, self.image_h, self.root_h):
            for col in range(0, self.image_w, self.root_w):
                descend(row, col, self.root_h, self.root_w)
        return sorted(result, key=lambda node: (-node[2], node[0], node[1]))

    def audit_leaves(self, mode='refine'):
        """Return one of two legal adjacent-scale audit partitions."""
        if mode not in {'refine', 'coarsen', 'finest'}:
            raise ValueError('audit mode must be refine, coarsen, or finest')
        if mode == 'finest':
            return [
                (row, col, MIN_H, MIN_W)
                for row in range(0, self.image_h, MIN_H)
                for col in range(0, self.image_w, MIN_W)
            ]
        if mode == 'refine':
            result = []
            for row, col, height, width in self.leaves():
                if (height, width) == (MIN_H, MIN_W):
                    result.append((row, col, height, width))
                    continue
                child_h, child_w = height // 2, width // 2
                for dr in (0, child_h):
                    for dc in (0, child_w):
                        result.append((row + dr, col + dc, child_h, child_w))
            return result

        result = []

        def descend(row, col, height, width):
            if self._is_leaf(row, col, height, width):
                result.append((row, col, height, width))
                return
            if self._children_are_leaves(row, col, height, width):
                result.append((row, col, height, width))
                return
            child_h, child_w = height // 2, width // 2
            for dr in (0, child_h):
                for dc in (0, child_w):
                    descend(row + dr, col + dc, child_h, child_w)

        for row in range(0, self.image_h, self.root_h):
            for col in range(0, self.image_w, self.root_w):
                descend(row, col, self.root_h, self.root_w)
        return result

    def required_evidence_levels(self):
        """Acquisition scales needed by the currently exposed decisions."""
        result = set()
        for row, col, height, width in self.decision_nodes():
            level = (height, width)
            child = (height // 2, width // 2)
            result.add(child)
            index = (row // height, col // width)
            if not self.split[level][index]:
                result.add(level)
        return result

    def update(self, train_histories, window_histories):
        train = {
            level: extract_scale_evidence(
                history,
                min_count=int(np.ceil(
                    self.min_train_count
                    * ((level[0] // MIN_H) * (level[1] // MIN_W))
                    ** self.count_scale_exponent
                )),
                support_threshold=self.support_threshold,
            )
            for level, history in train_histories.items()
        }
        window = prepare_pyramid(
            window_histories, min_count=0, support_threshold=-np.inf,
        )
        pending = {}
        for row, col, height, width in self.decision_nodes():
            level = (height, width)
            index = (row // height, col // width)
            child = (height // 2, width // 2)
            before_split = bool(self.split[level][index])
            # A parent decision is supported only by its four direct-child
            # group measurements.  A merged parent keeps receiving these via
            # the two refinement-audit batches, enabling later re-splitting.
            source_levels = (child,)
            evidence = []
            for source in source_levels:
                valid_i, n_reliable_i, n_required_i, gain_i, _ = (
                    predictive_plane_gain(
                        train[source], window[source], row, col, height, width,
                        source_level=source,
                        complexity_penalty=0.0,
                    )
                )
                if valid_i:
                    evidence.append((gain_i, n_reliable_i, n_required_i))
            valid = bool(evidence)
            if valid:
                gain = float(np.mean([item[0] for item in evidence]))
                n_reliable = max(item[1] for item in evidence)
                n_required = max(item[2] for item in evidence)
                delta = gain - self.complexity_penalty
            else:
                gain = delta = 0.0
                n_reliable = max(
                    predictive_plane_gain(
                        train[source], window[source], row, col, height, width,
                        source_level=source,
                        complexity_penalty=0.0,
                    )[1]
                    for source in source_levels
                )
                n_required = 4
            before_score = float(self.score[level][index])
            after_score = before_score
            if valid:
                after_score = float(np.clip(
                    self.rho * before_score + delta,
                    -self.score_cap,
                    self.score_cap,
                ))
                self.evidence_updates[level][index] += 1
                self.score[level][index] = after_score
            pending[(height, width, index[0], index[1])] = dict(
                node=(row, col, height, width), valid=valid,
                n_reliable=n_reliable, n_required=n_required,
                gain=gain, delta=delta, before_split=before_split,
                before_score=before_score,
            )

        # Penalized optimal tree cut.  Positive score is cost paid by keeping
        # a node coarse; negative score is cost paid by splitting it.  A split
        # adds three net leaves.  Unobserved nodes preserve their prior state,
        # which prevents either initialization from changing without evidence.
        choices = {}

        def best_cut(row, col, height, width):
            if (height, width) == (MIN_H, MIN_W):
                return 0.0, 0.0
            level = (height, width)
            index = (row // height, col // width)
            current = bool(self.split[level][index])
            observed = bool(self.evidence_updates[level][index] > 0)
            child_h, child_w = height // 2, width // 2
            child_cost = 0.0
            child_unresolved = 0.0
            for dr in (0, child_h):
                for dc in (0, child_w):
                    cost_i, unresolved_i = best_cut(
                        row + dr, col + dc, child_h, child_w
                    )
                    child_cost += cost_i
                    child_unresolved += unresolved_i
            if not observed:
                choices[(height, width, index[0], index[1])] = current
                return ((child_cost, child_unresolved)
                        if current else (0.0, child_unresolved))
            score = float(self.score[level][index])
            local_structure = max(score, 0.0)
            # Merging this node would hide every structured descendant.  Charge
            # that unresolved evidence to the coarse alternative so a smooth
            # parent cannot erase a child with a reliable split signal.
            unresolved = local_structure + child_unresolved
            keep_cost = unresolved
            scale_weight = ((self.root_h * self.root_w)
                            / float(height * width))
            split_cost = (max(-score, 0.0)
                          + 3.0 * self.leaf_penalty * scale_weight
                          + child_cost)
            if current:
                keep_cost += self.merge_threshold
            else:
                split_cost += self.split_threshold
            choose_split = split_cost < keep_cost
            choices[(height, width, index[0], index[1])] = choose_split
            return (split_cost if choose_split else keep_cost), unresolved

        for root_row in range(0, self.image_h, self.root_h):
            for root_col in range(0, self.image_w, self.root_w):
                best_cut(root_row, root_col, self.root_h, self.root_w)

        transitions = []

        def apply_visible(row, col, height, width):
            if (height, width) == (MIN_H, MIN_W):
                return
            key = (height, width, row // height, col // width)
            choose_split = choices[key]
            before = bool(self.split[(height, width)][key[2], key[3]])
            self.split[(height, width)][key[2], key[3]] = choose_split
            if before != choose_split:
                transitions.append((row, col, height, width,
                                    'split' if choose_split else 'merge'))
            if choose_split:
                child_h, child_w = height // 2, width // 2
                for dr in (0, child_h):
                    for dc in (0, child_w):
                        apply_visible(row + dr, col + dc, child_h, child_w)

        for root_row in range(0, self.image_h, self.root_h):
            for root_col in range(0, self.image_w, self.root_w):
                apply_visible(root_row, root_col, self.root_h, self.root_w)

        transition_by_node = {
            (height, width, row // height, col // width): transition
            for row, col, height, width, transition in transitions
        }
        updates = {}
        for key, item in pending.items():
            height, width, row_index, col_index = key
            level = (height, width)
            transition = transition_by_node.get(key, 'hold')
            after_split = bool(self.split[level][row_index, col_index])
            updates[(height, width, row_index, col_index)] = NodeUpdate(
                valid=item['valid'],
                n_reliable=item['n_reliable'],
                n_required=item['n_required'],
                predictive_gain=float(item['gain']),
                delta=float(item['delta']),
                score_before=item['before_score'],
                score_after=float(self.score[level][row_index, col_index]),
                state_before='split' if item['before_split'] else 'merged',
                state_after='split' if after_split else 'merged',
                transition=transition,
            )
        return updates, transitions

    def state_dict(self):
        result = {}
        for h, w in self.levels:
            tag = f'{h}x{w}'
            result[f'hier_split_{tag}'] = self.split[(h, w)]
            result[f'hier_score_{tag}'] = self.score[(h, w)]
            result[f'hier_evidence_updates_{tag}'] = self.evidence_updates[(h, w)]
        return result

    def load_state_dict(self, state):
        for h, w in self.levels:
            tag = f'{h}x{w}'
            for prefix, destination in (
                    ('hier_split_', self.split),
                    ('hier_score_', self.score),
                    ('hier_evidence_updates_', self.evidence_updates)):
                key = prefix + tag
                if key in state:
                    destination[(h, w)][...] = np.asarray(state[key])


class ResidualHierarchy(BidirectionalHierarchy):
    """v5-style 2/3 state machine driven by legal direct-child audits."""

    def __init__(self, *args, residual_thresholds=(0.50, 0.31, 0.29),
                 split_confirmations=2, merge_confirmations=3, **kwargs):
        super().__init__(*args, **kwargs)
        if len(residual_thresholds) != len(self.levels):
            raise ValueError('need one residual threshold per hierarchy level')
        self.residual_thresholds = {
            level: float(threshold)
            for level, threshold in zip(self.levels, residual_thresholds)
        }
        self.split_confirmations = int(split_confirmations)
        self.merge_confirmations = int(merge_confirmations)
        self.structured_streak = {
            level: np.zeros_like(self.split[level], dtype=np.int16)
            for level in self.levels
        }
        self.smooth_streak = {
            level: np.zeros_like(self.split[level], dtype=np.int16)
            for level in self.levels
        }

    @staticmethod
    def _combined_history(train, window):
        return {
            'a': np.asarray(train['a']) + np.asarray(window['a']),
            'b': np.asarray(train['b']) + np.asarray(window['b']),
            'na': np.asarray(train['na']) + np.asarray(window['na']),
            'nb': np.asarray(train['nb']) + np.asarray(window['nb']),
        }

    def update(self, train_histories, window_histories):
        required = {level for level in train_histories if level in window_histories}
        evidence = {
            level: extract_scale_evidence(
                self._combined_history(train_histories[level], window_histories[level]),
                min_count=int(np.ceil(
                    self.min_train_count
                    * ((level[0] // MIN_H) * (level[1] // MIN_W))
                    ** self.count_scale_exponent
                )),
                support_threshold=self.support_threshold,
            )
            for level in required
        }
        updates, transitions = {}, []
        for row, col, height, width in self.decision_nodes():
            level = (height, width)
            index = (row // height, col // width)
            child = (height // 2, width // 2)
            before_split = bool(self.split[level][index])
            before_score = float(self.score[level][index])
            if child in evidence:
                valid, n_reliable, n_required, residual = (
                    direct_child_plane_residual(
                        evidence[child], row, col, height, width, child
                    )
                )
            else:
                valid, n_reliable, n_required, residual = False, 0, 4, 0.0
            threshold = self.residual_thresholds[level]
            delta = residual - threshold if valid else 0.0
            transition = 'hold'
            if not valid:
                self.structured_streak[level][index] = 0
                self.smooth_streak[level][index] = 0
            elif delta > 0:
                self.structured_streak[level][index] += 1
                self.smooth_streak[level][index] = 0
                if (not before_split
                        and self.structured_streak[level][index]
                        >= self.split_confirmations):
                    self.split[level][index] = True
                    transition = 'split'
            else:
                self.smooth_streak[level][index] += 1
                self.structured_streak[level][index] = 0
                if (before_split
                        and self.smooth_streak[level][index]
                        >= self.merge_confirmations):
                    self.split[level][index] = False
                    transition = 'merge'
            if valid:
                self.score[level][index] = delta
                self.evidence_updates[level][index] += 1
            if transition != 'hold':
                transitions.append((row, col, height, width, transition))
            after_split = bool(self.split[level][index])
            updates[(height, width, index[0], index[1])] = NodeUpdate(
                valid=valid, n_reliable=n_reliable, n_required=n_required,
                predictive_gain=float(residual), delta=float(delta),
                score_before=before_score,
                score_after=float(self.score[level][index]),
                state_before='split' if before_split else 'merged',
                state_after='split' if after_split else 'merged',
                transition=transition,
            )
        return updates, transitions

    def state_dict(self):
        result = super().state_dict()
        for h, w in self.levels:
            tag = f'{h}x{w}'
            result[f'hier_structured_streak_{tag}'] = self.structured_streak[(h, w)]
            result[f'hier_smooth_streak_{tag}'] = self.smooth_streak[(h, w)]
        return result

    def load_state_dict(self, state):
        super().load_state_dict(state)
        for h, w in self.levels:
            tag = f'{h}x{w}'
            for prefix, destination in (
                    ('hier_structured_streak_', self.structured_streak),
                    ('hier_smooth_streak_', self.smooth_streak)):
                key = prefix + tag
                if key in state:
                    destination[(h, w)][...] = np.asarray(state[key])


class HybridHierarchy(ResidualHierarchy):
    """v5 global merge evidence plus direct-child evidence for re-splitting."""

    def _visible_nodes(self):
        result = []

        def descend(row, col, height, width):
            if (height, width) == (MIN_H, MIN_W):
                return
            result.append((row, col, height, width))
            level = (height, width)
            index = (row // height, col // width)
            if not self.split[level][index]:
                return
            child_h, child_w = height // 2, width // 2
            for dr in (0, child_h):
                for dc in (0, child_w):
                    descend(row + dr, col + dc, child_h, child_w)

        for row in range(0, self.image_h, self.root_h):
            for col in range(0, self.image_w, self.root_w):
                descend(row, col, self.root_h, self.root_w)
        return sorted(result, key=lambda node: (-node[2], node[0], node[1]))

    def update(self, train_histories, window_histories, global_history):
        required = {level for level in train_histories if level in window_histories}
        scale_evidence = {
            level: extract_scale_evidence(
                self._combined_history(train_histories[level], window_histories[level]),
                min_count=int(np.ceil(
                    self.min_train_count
                    * ((level[0] // MIN_H) * (level[1] // MIN_W))
                    ** self.count_scale_exponent
                )),
                support_threshold=self.support_threshold,
            )
            for level in required
        }
        global_evidence = extract_cell_evidence(
            global_history['a'], global_history['b'],
            int(global_history['na']), int(global_history['nb']),
        )
        updates, transitions = {}, []
        for row, col, height, width in self._visible_nodes():
            level = (height, width)
            index = (row // height, col // width)
            child = (height // 2, width // 2)
            before_split = bool(self.split[level][index])
            before_score = float(self.score[level][index])
            if before_split:
                decision = decide_node(
                    global_evidence, row // MIN_H, col // MIN_W,
                    height // MIN_H, width // MIN_W,
                )
                valid = decision.state != 'uncertain'
                n_reliable, n_required = decision.n_reliable, decision.n_required
                measurement_score = (
                    float(decision.n_plane) / float(decision.k_plane)
                    if valid else 0.0
                )
                delta = measurement_score - 1.0 if valid else 0.0
                structured = bool(decision.structured)
            else:
                if child in scale_evidence:
                    valid, n_reliable, n_required, measurement_score = (
                        direct_child_plane_residual(
                            scale_evidence[child], row, col, height, width, child
                        )
                    )
                else:
                    valid, n_reliable, n_required, measurement_score = (
                        False, 0, 4, 0.0
                    )
                delta = (measurement_score - self.residual_thresholds[level]
                         if valid else 0.0)
                structured = delta > 0

            transition = 'hold'
            if not valid:
                self.structured_streak[level][index] = 0
                self.smooth_streak[level][index] = 0
            elif structured:
                self.structured_streak[level][index] += 1
                self.smooth_streak[level][index] = 0
                if (not before_split
                        and self.structured_streak[level][index]
                        >= self.split_confirmations):
                    self.split[level][index] = True
                    transition = 'split'
            else:
                self.smooth_streak[level][index] += 1
                self.structured_streak[level][index] = 0
                if (before_split
                        and self.smooth_streak[level][index]
                        >= self.merge_confirmations):
                    self.split[level][index] = False
                    transition = 'merge'
            if valid:
                self.score[level][index] = delta
                self.evidence_updates[level][index] += 1
            if transition != 'hold':
                transitions.append((row, col, height, width, transition))
            after_split = bool(self.split[level][index])
            updates[(height, width, index[0], index[1])] = NodeUpdate(
                valid=valid, n_reliable=n_reliable, n_required=n_required,
                predictive_gain=float(measurement_score), delta=float(delta),
                score_before=before_score,
                score_after=float(self.score[level][index]),
                state_before='split' if before_split else 'merged',
                state_after='split' if after_split else 'merged',
                transition=transition,
            )
        return updates, transitions


class PixelDepthHierarchy(HybridHierarchy):
    """Bidirectional tree driven by decoded per-pixel depth structure.

    The depth and reliability maps must be decoded only from accumulated legal
    one-bit measurements.  Ground truth is never passed to this controller.
    A candidate block is structured when enough reliable pixel depths depart
    from its best affine plane, matching the oracle's geometric test directly.
    Separate split/merge fractions and confirmation streaks provide hysteresis.
    """

    def __init__(self, *args, pixel_residual_bins=4.0,
                 pixel_split_fraction=0.05, pixel_merge_fraction=0.02,
                 pixel_min_fraction=0.50, **kwargs):
        super().__init__(*args, **kwargs)
        if pixel_residual_bins <= 0:
            raise ValueError('pixel_residual_bins must be positive')
        if not 0 <= pixel_merge_fraction <= pixel_split_fraction <= 1:
            raise ValueError('need 0 <= merge fraction <= split fraction <= 1')
        if not 0 < pixel_min_fraction <= 1:
            raise ValueError('pixel_min_fraction must lie in (0, 1]')
        self.pixel_residual_bins = float(pixel_residual_bins)
        self.pixel_split_fraction = float(pixel_split_fraction)
        self.pixel_merge_fraction = float(pixel_merge_fraction)
        self.pixel_min_fraction = float(pixel_min_fraction)

    @staticmethod
    def _plane_outlier_fraction(depth, reliable, residual_bins):
        rows, cols = depth.shape
        n_reliable = int(np.count_nonzero(reliable))
        rr, cc = np.meshgrid(
            np.linspace(-1.0, 1.0, rows),
            np.linspace(-1.0, 1.0, cols), indexing='ij',
        )
        design = np.stack((np.ones_like(rr), rr, cc), axis=-1)
        theta, *_ = np.linalg.lstsq(
            design[reliable], depth[reliable], rcond=None
        )
        residual = np.abs(depth[reliable] - design[reliable] @ theta)
        return float(np.count_nonzero(residual > residual_bins)) / n_reliable

    def update(self, depth_bins, reliable_pixels):
        depth_bins = np.asarray(depth_bins, dtype=np.float32)
        reliable_pixels = np.asarray(reliable_pixels, dtype=bool)
        expected = (self.image_h, self.image_w)
        if depth_bins.shape != expected or reliable_pixels.shape != expected:
            raise ValueError(f'pixel maps must have shape {expected}')

        updates, transitions = {}, []
        for row, col, height, width in self._visible_nodes():
            level = (height, width)
            index = (row // height, col // width)
            before_split = bool(self.split[level][index])
            before_score = float(self.score[level][index])
            block_depth = depth_bins[row:row + height, col:col + width]
            block_reliable = reliable_pixels[
                row:row + height, col:col + width
            ]
            n_reliable = int(np.count_nonzero(block_reliable))
            n_required = max(
                4, int(np.ceil(self.pixel_min_fraction * height * width))
            )
            valid = n_reliable >= n_required
            outlier_fraction = (
                self._plane_outlier_fraction(
                    block_depth, block_reliable, self.pixel_residual_bins
                ) if valid else 0.0
            )
            delta = outlier_fraction - self.pixel_split_fraction if valid else 0.0
            transition = 'hold'
            if not valid:
                self.structured_streak[level][index] = 0
                self.smooth_streak[level][index] = 0
            elif outlier_fraction >= self.pixel_split_fraction:
                self.structured_streak[level][index] += 1
                self.smooth_streak[level][index] = 0
                if (not before_split
                        and self.structured_streak[level][index]
                        >= self.split_confirmations):
                    self.split[level][index] = True
                    transition = 'split'
            elif outlier_fraction <= self.pixel_merge_fraction:
                self.smooth_streak[level][index] += 1
                self.structured_streak[level][index] = 0
                if (before_split
                        and self.smooth_streak[level][index]
                        >= self.merge_confirmations):
                    self.split[level][index] = False
                    transition = 'merge'
            else:
                self.structured_streak[level][index] = 0
                self.smooth_streak[level][index] = 0

            if valid:
                self.score[level][index] = outlier_fraction
                self.evidence_updates[level][index] += 1
            if transition != 'hold':
                transitions.append((row, col, height, width, transition))
            after_split = bool(self.split[level][index])
            updates[(height, width, index[0], index[1])] = NodeUpdate(
                valid=valid, n_reliable=n_reliable, n_required=n_required,
                predictive_gain=float(outlier_fraction), delta=float(delta),
                score_before=before_score,
                score_after=float(self.score[level][index]),
                state_before='split' if before_split else 'merged',
                state_after='split' if after_split else 'merged',
                transition=transition,
            )
        return updates, transitions


class PixelEntropyHierarchy(HybridHierarchy):
    """Bidirectional tree driven only by raw decoded-depth entropy.

    No plane, gradient, residual, neighborhood, or smoothness statistic enters
    the decision.  Reliable A+B peak estimates are quantized into depth bins,
    and their empirical Shannon entropy is the sole split/merge score.
    """

    def __init__(self, *args, entropy_bin_width=4.0,
                 entropy_split_bits=(7.35, 6.27, 4.60),
                 entropy_merge_bits=(7.15, 6.07, 4.40),
                 pixel_min_fraction=0.50, **kwargs):
        super().__init__(*args, **kwargs)
        if entropy_bin_width <= 0:
            raise ValueError('entropy_bin_width must be positive')
        if not 0 < pixel_min_fraction <= 1:
            raise ValueError('pixel_min_fraction must lie in (0, 1]')

        def thresholds_by_level(values, name):
            values = np.atleast_1d(np.asarray(values, dtype=np.float64))
            if values.size == 1:
                values = np.repeat(values, len(self.levels))
            if values.size != len(self.levels):
                raise ValueError(
                    f'{name} needs one value or {len(self.levels)} level values'
                )
            return {
                level: float(value)
                for level, value in zip(self.levels, values)
            }

        self.entropy_bin_width = float(entropy_bin_width)
        self.entropy_split_bits = thresholds_by_level(
            entropy_split_bits, 'entropy_split_bits'
        )
        self.entropy_merge_bits = thresholds_by_level(
            entropy_merge_bits, 'entropy_merge_bits'
        )
        for level in self.levels:
            if not (0 <= self.entropy_merge_bits[level]
                    <= self.entropy_split_bits[level]):
                raise ValueError(
                    'need 0 <= entropy merge threshold <= split threshold '
                    f'at level {level}'
                )
        self.pixel_min_fraction = float(pixel_min_fraction)

    @staticmethod
    def _depth_entropy_bits(depth, reliable, bin_width):
        values = np.floor(depth[reliable] / bin_width).astype(np.int32)
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts.astype(np.float64) / float(counts.sum())
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def update(self, depth_bins, reliable_pixels):
        depth_bins = np.asarray(depth_bins, dtype=np.float32)
        reliable_pixels = np.asarray(reliable_pixels, dtype=bool)
        expected = (self.image_h, self.image_w)
        if depth_bins.shape != expected or reliable_pixels.shape != expected:
            raise ValueError(f'pixel maps must have shape {expected}')

        updates, transitions = {}, []
        for row, col, height, width in self._visible_nodes():
            level = (height, width)
            index = (row // height, col // width)
            before_split = bool(self.split[level][index])
            before_score = float(self.score[level][index])
            block_depth = depth_bins[row:row + height, col:col + width]
            block_reliable = reliable_pixels[
                row:row + height, col:col + width
            ]
            n_reliable = int(np.count_nonzero(block_reliable))
            n_required = max(
                4, int(np.ceil(self.pixel_min_fraction * height * width))
            )
            valid = n_reliable >= n_required
            entropy_bits = (
                self._depth_entropy_bits(
                    block_depth, block_reliable, self.entropy_bin_width
                ) if valid else 0.0
            )
            split_threshold = self.entropy_split_bits[level]
            merge_threshold = self.entropy_merge_bits[level]
            delta = entropy_bits - split_threshold if valid else 0.0
            transition = 'hold'
            if not valid:
                self.structured_streak[level][index] = 0
                self.smooth_streak[level][index] = 0
            elif entropy_bits >= split_threshold:
                self.structured_streak[level][index] += 1
                self.smooth_streak[level][index] = 0
                if (not before_split
                        and self.structured_streak[level][index]
                        >= self.split_confirmations):
                    self.split[level][index] = True
                    transition = 'split'
            elif entropy_bits <= merge_threshold:
                self.smooth_streak[level][index] += 1
                self.structured_streak[level][index] = 0
                if (before_split
                        and self.smooth_streak[level][index]
                        >= self.merge_confirmations):
                    self.split[level][index] = False
                    transition = 'merge'
            else:
                self.structured_streak[level][index] = 0
                self.smooth_streak[level][index] = 0

            if valid:
                self.score[level][index] = entropy_bits
                self.evidence_updates[level][index] += 1
            if transition != 'hold':
                transitions.append((row, col, height, width, transition))
            after_split = bool(self.split[level][index])
            updates[(height, width, index[0], index[1])] = NodeUpdate(
                valid=valid, n_reliable=n_reliable, n_required=n_required,
                predictive_gain=float(entropy_bits), delta=float(delta),
                score_before=before_score,
                score_after=float(self.score[level][index]),
                state_before='split' if before_split else 'merged',
                state_after='split' if after_split else 'merged',
                transition=transition,
            )
        return updates, transitions


def leaf_masks(leaves, image_h, image_w, levels):
    masks = {
        level: np.zeros((image_h // level[0], image_w // level[1]), dtype=bool)
        for level in levels
    }
    for row, col, height, width in leaves:
        masks[(height, width)][row // height, col // width] = True
    return masks


def cell_masks(leaves, image_h, image_w, levels):
    """3x4-cell masks keyed by the scale of the bit actually acquired."""
    shape = (image_h // MIN_H, image_w // MIN_W)
    masks = {level: np.zeros(shape, dtype=bool) for level in levels}
    for row, col, height, width in leaves:
        masks[(height, width)][
            row // MIN_H:(row + height) // MIN_H,
            col // MIN_W:(col + width) // MIN_W,
        ] = True
    return masks


def leaf_counts(leaves):
    counts = {(24, 32): 0, (12, 16): 0, (6, 8): 0, (3, 4): 0}
    for _, _, height, width in leaves:
        counts[(height, width)] = counts.get((height, width), 0) + 1
    return counts
