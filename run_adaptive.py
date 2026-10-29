"""Adaptive acquisition: the same chain, plus a controller that moves the partition.

Structurally this is run_nonadaptive.py with one feedback edge.  The group
partition is no longer a function of the batch index; it is chosen by a
controller reading evidence the decoder has already accumulated.  B is fixed for
the run, G evolves.

Every measurement stays legal.  A measured leaf returns one majority bit per
batch, exactly as in the non-adaptive case, and the controller sees only what
those bits imply -- never ground truth, which reconstruct.py loads after the
loop ends.

Exploration schedule
--------------------
Of every ``UPDATE_EVERY`` batches, the last ``AUDIT_BATCHES`` measure a forced
full-scene finest tiling instead of the current leaves.  Without this, a merged
region stops producing fine-scale evidence and can never be split again; the
audit keeps that evidence flowing at a fixed, declared cost in readout bits.

Two-half pixel evidence
-----------------------
Each pixel's own code is known, so a group's single bit backprojects to a
per-pixel contribution.  Those contributions accumulate into two never-reset
histories by batch parity (A for even batches, B for odd).  The depth peak is
taken from the combined A+B history, and a pixel counts as reliable only if BOTH
halves independently support that same peak.  One peak, two supports -- so a
peak produced by noise in one half does not qualify.

Controller variants, selected by SCALE and PIXEL_MEDIAN_SIZE:

    SCALE=pure      only pixels currently measured at the finest leaf size
                    contribute to the controller's histories
    SCALE=all       every measured pixel contributes, at whatever leaf size

The bilateral-profile and entropy-trigger variants from the earlier sweep are
deliberately NOT carried over here; add them back only if the comparison needs
them.

Usage:  run_adaptive.py [B]

A positional B on the command line is a convenience that sets the B
environment variable before running; every other setting is read from the
environment only, so B is consistent with the rest rather than a special
function argument.

Configuration is by environment variable; see run_nonadaptive.py for the shared
ones (SCENE, B, CODE, CODE_SHARING, P_DETECT, CYCLES, BILAT_*, VERIFY_BATCHES,
OUT_TAG) plus:

    ROOT                starting block, 24x32 (default) or 12x16
    START               finest (default) or coarsest
    UPDATE_EVERY        batches between controller updates, default 10
    AUDIT_BATCHES       audit batches per window, default 2 (even, < UPDATE_EVERY)
    SCALE               pure (default) or all
    PIXEL_MIN_COUNT     minimum accepted measurements per half, default 20
    PIXEL_SUPPORT       minimum per-half peak support, default 1.5
    PIXEL_MEDIAN_SIZE   median filter on decoded depth, default 3 (1 disables)
    PIXEL_RESIDUAL_BINS plane-residual outlier threshold in bins, default 4
    PIXEL_SPLIT_FRAC    outlier fraction that signals split, default 0.05
    PIXEL_MERGE_FRAC    outlier fraction that signals merge, default 0.02
    PIXEL_MIN_FRAC      reliable-pixel fraction needed to judge a block, default 0.50
    SPLIT_CONFIRM       consecutive split signals required, default 2
    MERGE_CONFIRM       consecutive merge signals required, default 3
"""
import os
import sys
import time

import cupy as cp
import numpy as np
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_config as dc

from adaptive_bidirectional_v6 import (
    MIN_H, MIN_W, PixelDepthHierarchy, cell_masks, leaf_counts,
)
from adaptive_partition import label_map

import checkpoint
import codes
import hw_delay
import decode
import encode
import photon_sample
import reconstruct


def configuration():
    """Read and validate the adaptive run configuration from the environment.
    Returns a dict of validated settings, including 'n_batches' and a
    nested 'bilateral' dict. AUDIT_BATCHES must be even so the A and B
    halves receive equal numbers of audit batches.
    """
    scene_name = os.environ.get('SCENE', 'bathroom-cycles-2')
    if scene_name not in dc.ALL_SCENES:
        raise ValueError(f'unknown scene {scene_name!r}')
    scene_cfg = dc.ALL_SCENES[scene_name]

    root = os.environ.get('ROOT', '24x32')
    if root not in ('24x32', '12x16'):
        raise ValueError(f"ROOT must be 24x32 or 12x16, got {root!r}")
    root_h, root_w = (int(part) for part in root.split('x'))

    start = os.environ.get('START', 'finest')
    if start not in ('finest', 'coarsest'):
        raise ValueError(f"START must be finest or coarsest, got {start!r}")

    update_every = int(os.environ.get('UPDATE_EVERY', 10))
    audit_batches = int(os.environ.get('AUDIT_BATCHES', 2))
    if update_every < 4:
        raise ValueError('UPDATE_EVERY must be at least 4')
    if audit_batches < 2 or audit_batches >= update_every or audit_batches % 2:
        raise ValueError('AUDIT_BATCHES must be even, at least 2, and < UPDATE_EVERY')

    scale = os.environ.get('SCALE', 'pure')
    if scale not in ('pure', 'all'):
        raise ValueError(f"SCALE must be pure or all, got {scale!r}")

    family = codes.resolve_family(os.environ.get('CODE', 'bernoulli'))

    p_detect = float(os.environ.get('P_DETECT', 1.0))
    if not 0.0 < p_detect <= 1.0:
        raise ValueError(f'P_DETECT must be in (0, 1], got {p_detect}')

    cycles = int(os.environ.get('CYCLES', dc.M))
    batch_size = int(os.environ.get('B', 1))
    if batch_size <= 0 or cycles % batch_size:
        raise ValueError(f'B={batch_size} must divide CYCLES={cycles}')

    # Hardware code-distribution delay (see hw_delay.py), off by default.
    # Format: "0:0.7,1:0.2,2:0.1" -- delay-in-bins:population-fraction, sum to 1.
    hw_delay_probs_raw = os.environ.get('HW_DELAY_PROBS')
    hw_delay_probs = None
    if hw_delay_probs_raw:
        hw_delay_probs = {}
        for part in hw_delay_probs_raw.split(','):
            delay_str, prob_str = part.split(':')
            hw_delay_probs[int(delay_str)] = float(prob_str)

    return dict(
        scene_cfg=scene_cfg, root_h=root_h, root_w=root_w, start=start,
        update_every=update_every, audit_batches=audit_batches, scale=scale,
        family=family, sharing=os.environ.get('CODE_SHARING', 'unshared'),
        p_detect=p_detect, hw_delay_probs=hw_delay_probs,
        batch_size=batch_size, cycles=cycles, n_batches=cycles // batch_size,
        verify_batches=int(os.environ.get('VERIFY_BATCHES', 0)),
        checkpoint_every=int(os.environ.get('CHECKPOINT_EVERY', 500)),
        out_tag=os.environ.get('OUT_TAG', ''),
        pixel_min_count=int(os.environ.get('PIXEL_MIN_COUNT', 20)),
        pixel_support=float(os.environ.get('PIXEL_SUPPORT', 1.5)),
        pixel_median_size=int(os.environ.get('PIXEL_MEDIAN_SIZE', 3)),
        pixel_residual_bins=float(os.environ.get('PIXEL_RESIDUAL_BINS', 4.0)),
        pixel_split_fraction=float(os.environ.get('PIXEL_SPLIT_FRAC', 0.05)),
        pixel_merge_fraction=float(os.environ.get('PIXEL_MERGE_FRAC', 0.02)),
        pixel_min_fraction=float(os.environ.get('PIXEL_MIN_FRAC', 0.50)),
        split_confirmations=int(os.environ.get('SPLIT_CONFIRM', 2)),
        merge_confirmations=int(os.environ.get('MERGE_CONFIRM', 3)),
        bilateral=dict(
            radius=int(os.environ.get('BILAT_RADIUS', reconstruct.BILATERAL_RADIUS)),
            sigma_spatial=float(os.environ.get(
                'BILAT_SIGMA_SPATIAL', reconstruct.BILATERAL_SIGMA_SPATIAL)),
            sigma_range=float(os.environ.get(
                'BILAT_SIGMA_RANGE', reconstruct.BILATERAL_SIGMA_RANGE)),
        ),
    )


def acquisition_config(cfg):
    """Settings that change the accumulated numbers or partition decisions,
    for checkpoint.fingerprint. Includes every controller threshold, since
    changing any of them changes the partition and so the whole run.
    Bilateral parameters are excluded (they affect only the final filter).
    """
    return dict(
        scene=cfg['scene_cfg'].name, method=method_tag(cfg),
        batch_size=cfg['batch_size'], scale=cfg['scale'],
        root=f"{cfg['root_h']}x{cfg['root_w']}", start=cfg['start'],
        code=cfg['family'], sharing=cfg['sharing'], p_detect=cfg['p_detect'],
        cycles=cfg['cycles'], decode='majority0',
        hw_delay_probs=cfg['hw_delay_probs'],
        update_every=cfg['update_every'], audit_batches=cfg['audit_batches'],
        pixel_min_count=cfg['pixel_min_count'],
        pixel_support=cfg['pixel_support'],
        pixel_median_size=cfg['pixel_median_size'],
        pixel_residual_bins=cfg['pixel_residual_bins'],
        pixel_split_fraction=cfg['pixel_split_fraction'],
        pixel_merge_fraction=cfg['pixel_merge_fraction'],
        pixel_min_fraction=cfg['pixel_min_fraction'],
        split_confirmations=cfg['split_confirmations'],
        merge_confirmations=cfg['merge_confirmations'],
    )


def method_tag(cfg):
    """Short identifier for the controller variant, e.g.
    'adaptive_scalepure_median' or 'adaptive_scaleall_nomedian'.
    """
    median = 'median' if cfg['pixel_median_size'] > 1 else 'nomedian'
    return f"adaptive_scale{cfg['scale']}_{median}"


def output_paths(cfg):
    """(result_path, partitions_path): result_path is an .npz with depth
    maps, MAEs and metadata; partitions_path is an .npz with the partition
    history.
    """
    scene_cfg = cfg['scene_cfg']
    stem = (f"{method_tag(cfg)}_B{cfg['batch_size']}_{cfg['family']}"
            f"_{cfg['sharing']}_p{cfg['p_detect']}_root{cfg['root_h']}x{cfg['root_w']}"
            f"_{cfg['start']}_majority0")
    if cfg['hw_delay_probs']:
        probs_str = '-'.join(f'{v}x{p:.2f}' for v, p in sorted(cfg['hw_delay_probs'].items()))
        stem += f'_hwdelay{probs_str}'
    stem += cfg['out_tag']
    directory = f'{dc.scene_sim_dir(scene_cfg)}/results'
    os.makedirs(directory, exist_ok=True)
    return f'{directory}/{stem}.npz', f'{directory}/partitions_{stem}.npz'


def fine_pixel_mask(leaves, R, Cw, levels):
    """(R*Cw,) bool device array: True where that pixel sits inside a 3x4
    leaf (measured at full resolution this batch), given the current
    partition's leaves and the hierarchy's levels.
    """
    masks = cell_masks(leaves, R, Cw, levels)
    cells = masks[(MIN_H, MIN_W)]
    # cells is one entry per 3x4 cell; expand to one entry per pixel.
    pixels = np.repeat(np.repeat(cells, MIN_H, axis=0), MIN_W, axis=1)
    return cp.asarray(pixels.reshape(-1))


def decoded_pixel_maps(history_a, history_b, count_a, count_b, cfg, R, Cw):
    """Combined-peak depth plus the two-half reliability mask.

    Args:
        history_a, history_b: (R*Cw, RES_T) float32 -- backprojected
            evidence from even/odd batches, never reset.
        count_a, count_b:     (R*Cw,) int32 -- accepted measurements per
            pixel in each half.
        cfg:   dict -- supplies pixel_min_count, pixel_support, pixel_median_size.
        R, Cw: image shape.

    Returns:
        depth:    (R, Cw) int host array -- estimated arrival bin per pixel
                  (median-filtered unless pixel_median_size is 1); what the
                  controller fits planes to.
        reliable: (R, Cw) bool host array -- True where the pixel has enough
                  measurements in both halves AND both halves independently
                  support the shared peak. Only these pixels may influence a
                  split or merge decision.
    """
    # One peak, chosen from the two halves pooled together -- picking a
    # peak separately per half and demanding agreement would be weaker,
    # since two noisy halves rarely agree exactly.
    combined = history_a + history_b
    bins = cp.argmax(combined, axis=-1).astype(cp.int16)

    def support_at_peak(history):
        """(R*Cw,) float z-score: how far this half's value at the shared
        peak stands above its own per-pixel mean/spread. 0 where flat."""
        selected = cp.take_along_axis(history, bins[:, None].astype(cp.int64), axis=-1)[:, 0]
        centre = cp.mean(history, axis=-1)
        scale = cp.std(history, axis=-1)
        return cp.where(scale > 1e-6, (selected - centre) / cp.maximum(scale, 1e-6),
                        cp.zeros_like(selected))

    # Trusted only if both halves have enough measurements AND both
    # independently support the shared peak -- rejects a peak created by
    # one half's noise, since the other half (disjoint batches) has no
    # reason to reproduce it.
    reliable = ((count_a >= cfg['pixel_min_count'])
                & (count_b >= cfg['pixel_min_count'])
                & (support_at_peak(history_a) >= cfg['pixel_support'])
                & (support_at_peak(history_b) >= cfg['pixel_support']))

    depth = cp.asnumpy(bins.reshape(R, Cw))
    if cfg['pixel_median_size'] > 1:
        depth = median_filter(depth, size=cfg['pixel_median_size'], mode='nearest')
    return depth, cp.asnumpy(reliable.reshape(R, Cw))


def main():
    """Run one adaptive acquisition from start (or a checkpoint) to end.
    Writes a result .npz, a partition-trace .npz, and periodic checkpoints;
    deletes the checkpoint on success. Ground truth is read only after the
    batch loop completes, so no partition decision can have seen it.
    Returns exit code 0 on success.
    """
    cfg = configuration()
    scene_cfg = cfg['scene_cfg']
    R, Cw, RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T

    # One residual threshold per hierarchy level; a 12x16 root has one
    # level fewer than a 24x32 one. Feeds the inherited residual machinery,
    # not the plane test this controller actually decides on.
    n_levels = 3 if (cfg['root_h'], cfg['root_w']) == (24, 32) else 2
    residual_thresholds = (0.50, 0.31, 0.29)[-n_levels:]

    hierarchy = PixelDepthHierarchy(
        R, Cw, cfg['root_h'], cfg['root_w'],
        initially_split=(cfg['start'] == 'finest'),
        residual_thresholds=residual_thresholds,
        support_threshold=cfg['pixel_support'],
        split_confirmations=cfg['split_confirmations'],
        merge_confirmations=cfg['merge_confirmations'],
        pixel_residual_bins=cfg['pixel_residual_bins'],
        pixel_split_fraction=cfg['pixel_split_fraction'],
        pixel_merge_fraction=cfg['pixel_merge_fraction'],
        pixel_min_fraction=cfg['pixel_min_fraction'],
    )
    levels = hierarchy.group_levels

    code_source = codes.CodeSource(
        scene_cfg, cfg['family'], sharing=cfg['sharing'])
    delay_source = (hw_delay.DelaySource(cfg['hw_delay_probs'])
                    if cfg['hw_delay_probs'] is not None else None)
    hw_delay_log = f" {hw_delay.describe(cfg['hw_delay_probs'])}" if cfg['hw_delay_probs'] else ''
    print(f"scene={scene_cfg.name} {method_tag(cfg)} B={cfg['batch_size']} "
          f"root={cfg['root_h']}x{cfg['root_w']} start={cfg['start']} "
          f"{code_source.describe()} p_detect={cfg['p_detect']} "
          f"update_every={cfg['update_every']} audit={cfg['audit_batches']} "
          f"median={cfg['pixel_median_size']} cycles={cfg['cycles']} "
          f"batches={cfg['n_batches']} decode=majority0{hw_delay_log}", flush=True)

    sampler = photon_sample.PhotonSampler(
        scene_cfg, cfg['batch_size'], cfg['p_detect'])
    accumulator = decode.new_accumulator(R, Cw, RES_T)
    raw_counts = decode.new_raw_counts(R, Cw, RES_T)
    history_a = cp.zeros((R * Cw, RES_T), dtype=cp.float32)
    history_b = cp.zeros((R * Cw, RES_T), dtype=cp.float32)

    # Depth-map and partition trace: one snapshot every dc.CKPT_INTERVAL
    # cycles, independent of change_batches/change_labels below (which
    # record only when the partition changed).
    assert cfg['cycles'] % dc.CKPT_INTERVAL == 0, \
        f"cycles={cfg['cycles']} must divide into dc.CKPT_INTERVAL={dc.CKPT_INTERVAL}-cycle snapshots"
    assert dc.CKPT_INTERVAL % cfg['batch_size'] == 0, \
        f"batch_size={cfg['batch_size']} must divide dc.CKPT_INTERVAL={dc.CKPT_INTERVAL}"
    n_ckpts = cfg['cycles'] // dc.CKPT_INTERVAL
    trace_raw = np.zeros((n_ckpts, R, Cw), dtype=np.float32)
    trace_recon = np.zeros((n_ckpts, R, Cw), dtype=np.float32)
    trace_bilat = np.zeros((n_ckpts, R, Cw), dtype=np.float32)
    trace_partition = np.zeros((n_ckpts, R, Cw), dtype=np.int16)
    trace_idx = 0
    count_a = cp.zeros(R * Cw, dtype=cp.int32)
    count_b = cp.zeros(R * Cw, dtype=cp.int32)

    def refresh_partition():
        """Rebuild both measurement partitions (current and audit) from the
        controller's current tree. Returns a dict with, for each: leaves,
        flat label map, group count, and finest-scale pixel mask --
        'current_leaves', 'current_labels', 'current_n', 'current_fine',
        'audit_leaves', 'audit_labels', 'audit_n', 'audit_fine'. Called at
        startup and again after every controller update.
        """
        current = hierarchy.leaves()
        audit = hierarchy.audit_leaves('finest')
        return dict(
            current_leaves=current,
            current_labels=encode.flat_labels(np.asarray(label_map(current, R, Cw), np.int32)),
            current_n=len(current),
            current_fine=fine_pixel_mask(current, R, Cw, levels),
            audit_leaves=audit,
            audit_labels=encode.flat_labels(np.asarray(label_map(audit, R, Cw), np.int32)),
            audit_n=len(audit),
            audit_fine=fine_pixel_mask(audit, R, Cw, levels),
        )

    stop_batch = (min(cfg['n_batches'], cfg['verify_batches'])
                  if cfg['verify_batches'] else cfg['n_batches'])
    if stop_batch != cfg['n_batches']:
        print(f'VERIFICATION MODE: stopping after batch {stop_batch}, '
              'ground truth will not be loaded', flush=True)

    result_path, partitions_path = output_paths(cfg)
    checkpoint_path = checkpoint.path_for(result_path)
    acquisition = acquisition_config(cfg)
    total_readout_bits = 0
    start_batch = 0

    state = checkpoint.load(checkpoint_path, acquisition)
    if state is not None:
        start_batch = int(state['next_batch'])
        accumulator = cp.asarray(state['accumulator'])
        raw_counts = cp.asarray(state['raw_counts'])
        history_a = cp.asarray(state['history_a'])
        history_b = cp.asarray(state['history_b'])
        count_a = cp.asarray(state['count_a'])
        count_b = cp.asarray(state['count_b'])
        total_readout_bits = int(state['total_readout_bits'])
        sampler.accepted_by_category = state['accepted_by_category'].copy()
        sampler.rejected = int(state['rejected_photons'])
        trace_raw = state['trace_raw']
        trace_recon = state['trace_recon']
        trace_bilat = state['trace_bilat']
        trace_partition = state['trace_partition']
        trace_idx = int(state['trace_idx'])
        hierarchy.load_state_dict(state)  # so resume continues the partition history
        change_batches = [int(x) for x in state['change_batch']]
        change_labels = [np.asarray(x, np.int16) for x in state['change_labels']]
        label_image = np.asarray(state['label_image'], np.int16)
        partition = refresh_partition()
        print(f'RESUMING from batch {start_batch} with '
              f"{partition['current_n']} leaves; replaying RNG streams",
              flush=True)
        reposition_started = time.time()
        replayed = checkpoint.reposition_streams(
            sampler, code_source, start_batch, labels_for_batch=None)
        if delay_source is not None:
            delay_source.fast_forward(start_batch, R * Cw, cfg['batch_size'])
        print(f'  repositioned ({replayed} batches replayed, '
              f'{time.time() - reposition_started:.0f}s)', flush=True)
    else:
        partition = refresh_partition()
        label_image = np.asarray(label_map(partition['current_leaves'], R, Cw), np.int16)
        change_batches = [0]
        change_labels = [label_image.copy()]

    started = time.time()

    def write_checkpoint(next_batch):
        """Persist everything accumulated so far, including the controller
        state (next_batch: where a resumed run should start). ~2.5 GB:
        accumulator, raw counts, both pixel histories, hierarchy state and
        partition trace -- without the hierarchy state, resume would
        silently restart from the initial tree.
        """
        checkpoint.save(
            checkpoint_path, next_batch, acquisition,
            dict(accumulator=cp.asnumpy(accumulator),
                 raw_counts=cp.asnumpy(raw_counts),
                 history_a=cp.asnumpy(history_a),
                 history_b=cp.asnumpy(history_b),
                 count_a=cp.asnumpy(count_a),
                 count_b=cp.asnumpy(count_b),
                 total_readout_bits=np.int64(total_readout_bits),
                 accepted_by_category=sampler.accepted_by_category,
                 rejected_photons=np.int64(sampler.rejected),
                 change_batch=np.asarray(change_batches, np.int32),
                 change_labels=np.stack(change_labels).astype(np.int16),
                 label_image=label_image,
                 trace_raw=trace_raw, trace_recon=trace_recon,
                 trace_bilat=trace_bilat, trace_partition=trace_partition,
                 trace_idx=np.int64(trace_idx),
                 **hierarchy.state_dict()))

    for batch in range(start_batch, stop_batch):
        offset = batch % cfg['update_every']

        # The last AUDIT_BATCHES of each window force a full-scene finest
        # tiling instead of the current leaves -- the exploration schedule:
        # a merged (coarse) region stops producing fine-scale evidence, so
        # without these audits it could never be split again.
        use_audit = offset >= cfg['update_every'] - cfg['audit_batches']
        if use_audit:
            labels_flat, n_groups = partition['audit_labels'], partition['audit_n']
            fine_pixels = partition['audit_fine']
        else:
            labels_flat, n_groups = partition['current_labels'], partition['current_n']
            fine_pixels = partition['current_fine']

        # Same five stages as the non-adaptive driver; only the partition
        # differs, and it is still one legal bit per measured group.
        code_pb = code_source.draw(batch, labels_flat, n_groups)
        arrivals = sampler.sample(batch)
        decode.accumulate_raw(raw_counts, arrivals, RES_T)
        delay_bins = (delay_source.draw(R * Cw, cfg['batch_size'])
                      if delay_source is not None else None)
        bits = encode.group_bits(code_pb, labels_flat, n_groups, arrivals,
                                  pixel_delay_bins=delay_bins)

        # Inlined rather than decode.backproject since `contribution` is
        # needed twice: the reconstruction accumulator, and the
        # controller's evidence below.
        contribution = code_pb.astype(cp.float32)
        contribution *= bits[labels_flat][:, None]
        accumulator += contribution
        total_readout_bits += n_groups

        # Controller evidence. batch % 2 is the A/B split (even -> history
        # A, odd -> history B); the halves are never reset or mixed, so
        # they're two independent views, which is what makes the two-half
        # reliability test in decoded_pixel_maps meaningful.
        if cfg['scale'] == 'all':
            # Every measured pixel contributes, whatever leaf size it was
            # measured in -- more evidence, but coarse leaves blur.
            if batch % 2 == 0:
                history_a += contribution
                count_a += 1
            else:
                history_b += contribution
                count_b += 1
        else:
            # Only pixels measured at the finest (3x4) size count -- less
            # evidence, but all of it at full resolution.
            weight = fine_pixels[:, None]
            if batch % 2 == 0:
                history_a += cp.where(weight, contribution, 0.0)
                count_a += fine_pixels.astype(cp.int32)
            else:
                history_b += cp.where(weight, contribution, 0.0)
                count_b += fine_pixels.astype(cp.int32)
        del contribution, code_pb, bits, arrivals

        completed = batch + 1
        cycles_done = completed * cfg['batch_size']
        if cycles_done % dc.CKPT_INTERVAL == 0:
            maps_now = reconstruct.depth_maps(accumulator, raw_counts, scene_cfg, **cfg['bilateral'])
            trace_raw[trace_idx] = maps_now['raw']
            trace_recon[trace_idx] = maps_now['reconstructed']
            trace_bilat[trace_idx] = maps_now['bilateral']
            trace_partition[trace_idx] = label_image
            trace_idx += 1

        # The feedback edge: every UPDATE_EVERY batches, decode what has
        # accumulated and let the controller reshape the partition. It
        # fits a plane to each block's reliable depths and splits/merges
        # on how many pixels depart from it -- sees only decoded
        # measurements, never ground truth.
        if completed % cfg['update_every'] == 0:
            depth, reliable = decoded_pixel_maps(
                history_a, history_b, count_a, count_b, cfg, R, Cw)
            updates, _ = hierarchy.update(depth, reliable)
            partition = refresh_partition()
            new_labels = np.asarray(label_map(partition['current_leaves'], R, Cw), np.int16)
            # Record only when the partition actually changed, so the
            # saved trace is a list of transitions, not one frame/update.
            if not np.array_equal(new_labels, label_image):
                label_image = new_labels
                change_batches.append(completed)
                change_labels.append(label_image.copy())
            counts = leaf_counts(partition['current_leaves'])
            splits = sum(u.transition == 'split' for u in updates.values())
            merges = sum(u.transition == 'merge' for u in updates.values())
            valid = sum(int(u.valid) for u in updates.values())
            print(f"  batch {completed}/{stop_batch}: leaves={partition['current_n']} "
                  f"[24x32={counts[(24, 32)]}, 12x16={counts[(12, 16)]}, "
                  f"6x8={counts[(6, 8)]}, 3x4={counts[(3, 4)]}] "
                  f"valid={valid}/{len(updates)} split={splits} merge={merges} "
                  f"bits={total_readout_bits} ({time.time() - started:.0f}s)", flush=True)

        if cfg['checkpoint_every'] and completed % cfg['checkpoint_every'] == 0 \
                and completed != stop_batch:
            write_checkpoint(completed)

    np.savez_compressed(
        partitions_path,
        adaptive_method=np.array(method_tag(cfg)),
        scene=np.array(scene_cfg.name),
        start_mode=np.array(cfg['start']),
        root_block=np.array([cfg['root_h'], cfg['root_w']], np.int16),
        change_batch=np.asarray(change_batches, np.int32),
        change_labels=np.stack(change_labels).astype(np.int16),
        final_labels=label_image,
        total_readout_bits=np.int64(total_readout_bits),
        # Dense, fixed-cadence trace (one frame per CKPT_INTERVAL, aligned
        # with trace_raw/recon/bilat), unlike change_batch/change_labels.
        partition_trace=trace_partition,
    )

    if stop_batch != cfg['n_batches']:
        write_checkpoint(stop_batch)
        print(f'VERIFIED through batch {stop_batch}; resumable checkpoint saved; '
              'ground truth was never loaded', flush=True)
        return 0

    maps = dict(raw=trace_raw, reconstructed=trace_recon, bilateral=trace_bilat)
    gt = reconstruct.load_ground_truth(scene_cfg)
    errors = {name: reconstruct.mae(trace[-1], gt) for name, trace in maps.items()}
    metadata = dict(
        scene=scene_cfg.name, adaptive_method=method_tag(cfg),
        batch_size=cfg['batch_size'], scale=cfg['scale'],
        start_mode=cfg['start'], root_block=[cfg['root_h'], cfg['root_w']],
        code=cfg['family'], code_sharing=cfg['sharing'],
        p_detect=cfg['p_detect'], cycles=cfg['cycles'],
        n_batches=cfg['n_batches'], decode='majority0',
        update_every=cfg['update_every'], audit_batches=cfg['audit_batches'],
        pixel_median_size=cfg['pixel_median_size'],
        total_readout_bits=total_readout_bits,
        final_leaves=int(len(partition['current_leaves'])),
        accepted_by_category=sampler.accepted_by_category,
        category_names=list(photon_sample.CATEGORIES),
        rejected_photons=sampler.rejected,
    )
    path = reconstruct.save(result_path, maps, gt, errors, metadata)
    checkpoint.discard(checkpoint_path)
    print('MAE (cm): raw / recon / bilat = '
          f"{errors['raw']:.2f} / {errors['reconstructed']:.2f} / "
          f"{errors['bilateral']:.2f}", flush=True)
    print(f'wrote {path}', flush=True)
    print(f'wrote {partitions_path}', flush=True)
    print('Done.', flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        os.environ['B'] = sys.argv[1]
    sys.exit(main())
