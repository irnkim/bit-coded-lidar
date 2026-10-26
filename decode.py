"""Turns the measured bits back into a per-pixel estimate.

encode.py produces one bit per group. Every pixel in that group knows its own
temporal code, so each backprojects the group's single bit through its own
code with no further information from the sensor:

    accumulator[pixel, t] += y_group(pixel) * code[pixel, t]

A group that saw no photon emitted 0, so it contributes nothing.

Also accumulates the uncoded photon counts -- costs nothing, is independent
of the coding, and gives the raw-transient baseline for free.
"""
import cupy as cp
import cupyx


def new_accumulator(R, Cw, RES_T):
    """(R*Cw, RES_T) float32 zeros -- the coded-evidence accumulator."""
    return cp.zeros((R * Cw, RES_T), dtype=cp.float32)


def new_raw_counts(R, Cw, RES_T):
    """(R*Cw*RES_T,) int32 zeros, flat (scatter_add needs a 1-D destination)."""
    return cp.zeros(R * Cw * RES_T, dtype=cp.int32)


def accumulate_raw(raw_counts, arrivals, RES_T):
    """Add this batch's detected photons to the uncoded baseline histogram,
    in place. arrivals: (n_pixels, batch_size) int, arrival bin per
    (pixel, cycle) slot or -1 for no detection that cycle.
    """
    n_pixels = arrivals.shape[0]
    valid = arrivals >= 0
    pixel_index = cp.broadcast_to(
        cp.arange(n_pixels, dtype=cp.int64)[:, None], arrivals.shape)
    flat_index = pixel_index[valid] * RES_T + arrivals[valid].astype(cp.int64)
    cupyx.scatter_add(raw_counts, flat_index, 1)


def backproject(accumulator, code_pb, bits, labels_flat):
    """Add this batch's contribution to accumulator, in place: each pixel's
    own code times its group's bit. code_pb must be the same per-pixel codes
    passed to encode.group_bits for this batch, and bits/labels_flat its
    output -- a mismatched code decodes against the wrong basis.
    """
    contribution = code_pb.astype(cp.float32)
    # bits[labels_flat] fans each group's one bit out to all its pixels;
    # multiplying by the pixel's own code is the decode step -- agreement in
    # sign accumulates positive evidence at the true arrival bin.
    contribution *= bits[labels_flat][:, None]
    accumulator += contribution


def estimate_bins(accumulator, R, Cw):
    """(R, Cw) int host array: argmax of accumulated coded evidence, the
    per-pixel depth estimate in time bins.
    """
    return cp.asnumpy(accumulator.argmax(axis=-1)).reshape(R, Cw)


def raw_bins(raw_counts, R, Cw, RES_T):
    """(R, Cw) int host array: argmax of the uncoded photon histogram, the
    baseline a per-pixel-TDC sensor would give.
    """
    return cp.asnumpy(raw_counts.reshape(R, Cw, RES_T).argmax(axis=-1))
