"""The sensor model: how pixels are grouped, and the one bit each group emits.

A group's member pixels share one physical detector, so no matter how many
photons the group collects in a batch, exactly ONE bit comes down the wire:

    y_g = sign( sum over photons in group g of that photon's own code value )

with ``sign(0) = 0`` -- majority vote, zero on a tie. That's the only
decoder; a signed photon sum is not a legal sensor output.

Backprojecting y into a depth estimate is decode.py's job, not this module's.

``group_labels`` covers every configuration with one implementation; a fixed
grid is just ``shift_step=0``:

  G = (1, 1)                  every pixel its own group (transient/temporal)
  G > (1, 1), shift_step=0    fixed block, lattice never moves
  G > (1, 1), shift_step=1    sliding block, advances one pixel per batch

Sliding advances per BATCH, not per pulse -- a lattice moving mid-batch would
leave a pixel in several groups for one readout bit. A shifted lattice has
slightly more groups than an aligned one (edge fragments), so readout-bit
count must be reported rather than assumed equal across shift_step. Edges
never wrap: a shifted lattice's leading/trailing fragments become their own
smaller groups rather than joining spatially distant pixels.
"""
import cupy as cp
import cupyx
import numpy as np


def segments_no_wrap(length, size, shift):
    """Partition [0, length) into contiguous segments under a shifted lattice
    (splits rather than wraps at the edge).

    Returns:
        ids:    (length,) int32 -- segment number per index, 0..n_segs-1.
        n_segs: int -- number of segments (length // size at shift=0; one or
                two more when shift is nonzero).

    Example: length=10, size=3, shift=2 -> ids=[0,0,1,1,1,2,2,2,3,3], n_segs=4
    """
    if not 0 <= shift < size:
        raise ValueError(f'shift {shift} out of range for size {size}')

    ids = np.full(length, -1, dtype=np.int32)
    next_id = 0

    if shift > 0:
        ids[0:shift] = next_id
        next_id += 1

    start = shift
    while start + size <= length:
        ids[start:start + size] = next_id
        next_id += 1
        start += size

    if start < length:
        ids[start:length] = next_id
        next_id += 1

    return ids, next_id


def partition_shift(batch_index, g_h, g_w, shift_step=0):
    """The lattice offset for a batch: (shift_r, shift_c) in [0, g_h) x
    [0, g_w) to pass to group_labels. shift_step=0 (fixed block) always
    gives (0, 0); shift_step=n advances the lattice n pixels per batch.

    Example: batch 7, g=(3,4), shift_step=1 -> (1, 3)   # 7%3=1, 7%4=3
    """
    if shift_step < 0:
        raise ValueError(f'shift_step must be non-negative, got {shift_step}')
    offset = batch_index * shift_step
    return offset % g_h, offset % g_w


def group_labels(R, Cw, g_h, g_w, shift_r=0, shift_c=0):
    """Assign every pixel to a readout group (rectangular tiles = the outer
    product of two independent 1-D partitions, rows x cols).

    Returns:
        labels:   (R, Cw) int32 -- group id per pixel, 0..n_groups-1.
        n_groups: int -- number of groups, i.e. readout bits per batch.

    Example: R=240, Cw=320, g=(3,4), no shift -> n_groups=6400;
             g=(1,1) -> labels == flat pixel index, n_groups=76800.
    """
    row_ids, n_row = segments_no_wrap(R, g_h, shift_r)
    col_ids, n_col = segments_no_wrap(Cw, g_w, shift_c)
    labels = row_ids[:, None] * n_col + col_ids[None, :]
    return labels.astype(np.int32), int(n_row) * int(n_col)


def flat_labels(labels):
    """(R*Cw,) int32 device array: labels moved to GPU and flattened to
    match photons' flat pixel index (row * Cw + col).
    """
    return cp.asarray(np.ascontiguousarray(labels).reshape(-1))


def group_bits(code_pb, labels_flat, n_groups, arrivals, pixel_delay_bins=None):
    """The sole legal output per group: sign(sum of detected photons' own codes).

    Args:
        code_pb:     (R*Cw, RES_T) int8 in {-1, +1} -- each pixel's own code,
                     held constant for this batch.
        labels_flat: (R*Cw,) int32 -- group id per pixel.
        n_groups:    int -- output length; must match labels_flat.
        arrivals:    (R*Cw, batch_size) int, from PhotonSampler.sample --
                     arrival bin per (pixel, cycle) slot, or -1 for none.
        pixel_delay_bins: (R*Cw, batch_size) int32 or None -- hardware
                     code-distribution delay in time bins (see hw_delay.py).
                     When given, each photon's vote reads code_pb at
                     (arrival_bin - delay), a stale sample, while
                     decode.backproject keeps correlating against the true
                     code -- the encode/decode mismatch under study. None
                     reproduces the undelayed behavior exactly.

    Returns:
        (n_groups,) float32, each entry in {-1, 0, +1}: +1/-1 by majority,
        0 on an exact tie or if the group detected nothing (contributes
        nothing once backprojected).
    """
    n_pixels = arrivals.shape[0]
    valid = arrivals >= 0
    pixel_index = cp.broadcast_to(
        cp.arange(n_pixels, dtype=cp.int64)[:, None], arrivals.shape)
    event_pixels = pixel_index[valid]
    event_bins = arrivals[valid].astype(cp.int64)
    event_labels = labels_flat[event_pixels]

    # Each photon's vote: its own pixel's code value at its arrival bin (or,
    # under a hardware delay, at the bin the lagging code signal still held).
    sample_bins = event_bins
    if pixel_delay_bins is not None:
        cycle_index = cp.broadcast_to(
            cp.arange(arrivals.shape[1], dtype=cp.int64)[None, :], arrivals.shape)
        event_cycles = cycle_index[valid]
        event_delays = pixel_delay_bins[event_pixels, event_cycles]
        sample_bins = cp.clip(event_bins - event_delays, 0, code_pb.shape[1] - 1)
    own_sign = code_pb[event_pixels, sample_bins]

    # scatter_add: parallel-safe "+= own_sign at this group" when many
    # photons land in the same group at once.
    vote_sum = cp.zeros(n_groups, dtype=cp.int32)
    cupyx.scatter_add(vote_sum, event_labels, own_sign.astype(cp.int32))
    return cp.sign(vote_sum).astype(cp.float32)


def group_bits_hierarchical(code_pb, labels_flat, n_groups, arrivals):
    """Majority of per-cycle majorities, instead of one sum over the whole
    batch (hardware/CIRCUITS.md Step 7's ACC2, vs. group_bits' Step 4 ACC).
    Retains ~98-100% of group_bits' evidence on the same photon model, since
    ACC2 is itself exact. Same args/return convention as group_bits.
    """
    n_pixels, batch_size = arrivals.shape
    pixel_index = cp.arange(n_pixels, dtype=cp.int64)
    cycle_acc = cp.zeros(n_groups, dtype=cp.int32)

    for c in range(batch_size):
        arrivals_c = arrivals[:, c]
        valid = arrivals_c >= 0
        event_pixels = pixel_index[valid]
        event_bins = arrivals_c[valid].astype(cp.int64)
        event_labels = labels_flat[event_pixels]
        own_sign = code_pb[event_pixels, event_bins]

        d_cycle = cp.zeros(n_groups, dtype=cp.int32)
        cupyx.scatter_add(d_cycle, event_labels, own_sign.astype(cp.int32))
        cycle_acc += cp.sign(d_cycle).astype(cp.int32)

    return cp.sign(cycle_acc).astype(cp.float32)
