"""Non-adaptive acquisition: one driver for every fixed-partition configuration.

The G x B plane covers four dimensions with no special cases:

    G = 1x1, B = 1          transient
    G = 1x1, B > 1          temporal
    G > 1x1, B = 1          spatial
    G > 1x1, B > 1          spatial + temporal

and ``SHIFT_STEP`` selects a fixed block (0) or a sliding one (1).

Usage:  run_nonadaptive.py [B]

A positional B on the command line is a convenience that sets the B
environment variable before running; every other setting is read from the
environment only, so B is consistent with the rest rather than a special
function argument.

Configuration is by environment variable:

    SCENE               bathroom-cycles-2 (default) or any dataset_config scene
    B                    cycles per batch, default 1
    G_H, G_W            group height/width, default 1 1
    SHIFT_STEP          0 fixed block (default), 1 sliding block
    CODE                bernoulli (default), lowpass, trig, chipoffset --
                        also accepts filtered-bernoulli/cosine-wave/
                        wide-bernoulli as aliases for lowpass/trig/
                        chipoffset (see codes.FAMILY_ALIASES); resolved to
                        the canonical name before it touches a filename
    CODE_SHARING        unshared (default) or shared
    P_DETECT            signal detection probability in (0, 1], default 1.0
    HW_DELAY_MODE        per_cycle (default) or per_batch_group -- see
                        hw_delay.DelaySource.draw_per_group
    CYCLES              illumination cycles, default dataset_config.M
    BILAT_RADIUS        default 5
    BILAT_SIGMA_SPATIAL default 2.0
    BILAT_SIGMA_RANGE   default 0.15
    VERIFY_BATCHES      stop after this many batches without evaluating, for
                        smoke tests -- ground truth is never loaded
    OUT_TAG             suffix appended to the output filename

Ground truth is read only after the full batch loop finishes, inside
reconstruct.py.  A run stopped early by VERIFY_BATCHES returns before that.
"""
import os
import sys
import time

import cupy as cp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_config as dc

import checkpoint
import codes
import hw_delay
import decode
import encode
import photon_sample
import reconstruct


def configuration():
    """Read and validate the run configuration from the environment.
    Returns a dict of validated settings, including 'n_batches'
    (cycles // B) and a nested 'bilateral' dict. Raises ValueError on an
    unknown scene, a group larger than the image, a B that doesn't divide
    the cycle budget, or a P_DETECT outside (0, 1].
    """
    scene_name = os.environ.get('SCENE', 'bathroom-cycles-2')
    if scene_name not in dc.ALL_SCENES:
        raise ValueError(f'unknown scene {scene_name!r}')
    scene_cfg = dc.ALL_SCENES[scene_name]

    g_h = int(os.environ.get('G_H', 1))
    g_w = int(os.environ.get('G_W', 1))
    if not (1 <= g_h <= scene_cfg.R and 1 <= g_w <= scene_cfg.Cw):
        raise ValueError(f'group {g_h}x{g_w} does not fit {scene_cfg.R}x{scene_cfg.Cw}')

    shift_step = int(os.environ.get('SHIFT_STEP', 0))
    family = codes.resolve_family(os.environ.get('CODE', 'bernoulli'))
    sharing = os.environ.get('CODE_SHARING', 'unshared')

    p_detect = float(os.environ.get('P_DETECT', 1.0))
    if not 0.0 < p_detect <= 1.0:
        raise ValueError(f'P_DETECT must be in (0, 1], got {p_detect}')
    cycles = int(os.environ.get('CYCLES', dc.M))
    batch_size = int(os.environ.get('B', 1))
    if batch_size <= 0 or cycles % batch_size:
        raise ValueError(f'B={batch_size} must divide CYCLES={cycles}')

    # majority0: sign of the sum over the whole batch (encode.group_bits).
    # hier0: sign per cycle first, those trits summed across the batch
    # (encode.group_bits_hierarchical).
    decode_mode = os.environ.get('DECODE_MODE', 'majority0')
    if decode_mode not in ('majority0', 'hier0'):
        raise ValueError(f'DECODE_MODE must be majority0 or hier0, got {decode_mode!r}')

    # Hardware code-distribution delay (see hw_delay.py), off by default.
    # Format: "0:0.7,1:0.2,2:0.1" -- delay-in-bins:population-fraction, sum to 1.
    hw_delay_probs_raw = os.environ.get('HW_DELAY_PROBS')
    hw_delay_probs = None
    if hw_delay_probs_raw:
        hw_delay_probs = {}
        for part in hw_delay_probs_raw.split(','):
            delay_str, prob_str = part.split(':')
            hw_delay_probs[int(delay_str)] = float(prob_str)

    # per_cycle (default): independent delay draw per (pixel, cycle).
    # per_batch_group: one draw per (batch, group) -- every pixel in a
    # group and every cycle in the batch shares the same delay (see
    # hw_delay.DelaySource.draw_per_group).
    hw_delay_mode = os.environ.get('HW_DELAY_MODE', 'per_cycle')
    if hw_delay_mode not in ('per_cycle', 'per_batch_group'):
        raise ValueError(f'HW_DELAY_MODE must be per_cycle or per_batch_group, got {hw_delay_mode!r}')

    return dict(
        scene_cfg=scene_cfg, g_h=g_h, g_w=g_w, shift_step=shift_step,
        family=family, sharing=sharing, p_detect=p_detect,
        batch_size=batch_size, cycles=cycles, decode_mode=decode_mode,
        hw_delay_probs=hw_delay_probs, hw_delay_mode=hw_delay_mode,
        n_batches=cycles // batch_size,
        verify_batches=int(os.environ.get('VERIFY_BATCHES', 0)),
        checkpoint_every=int(os.environ.get('CHECKPOINT_EVERY', 500)),
        out_tag=os.environ.get('OUT_TAG', ''),
        bilateral=dict(
            radius=int(os.environ.get('BILAT_RADIUS', reconstruct.BILATERAL_RADIUS)),
            sigma_spatial=float(os.environ.get(
                'BILAT_SIGMA_SPATIAL', reconstruct.BILATERAL_SIGMA_SPATIAL)),
            sigma_range=float(os.environ.get(
                'BILAT_SIGMA_RANGE', reconstruct.BILATERAL_SIGMA_RANGE)),
        ),
    )


def acquisition_config(cfg):
    """Settings that change the accumulated numbers, for checkpoint.fingerprint.
    Bilateral parameters are deliberately excluded: they affect only the
    final filter, so changing them must not invalidate a checkpoint.
    """
    return dict(
        scene=cfg['scene_cfg'].name, g_h=cfg['g_h'], g_w=cfg['g_w'],
        batch_size=cfg['batch_size'], shift_step=cfg['shift_step'],
        code=cfg['family'], sharing=cfg['sharing'], p_detect=cfg['p_detect'],
        cycles=cfg['cycles'], decode=cfg['decode_mode'],
        hw_delay_probs=cfg['hw_delay_probs'], hw_delay_mode=cfg['hw_delay_mode'],
    )


def shared_labels_for(cfg, R, Cw, batch):
    """The group labels a given batch used, needed to replay shared-mode
    code draws (the draw shape depends on that batch's group count).
    Returns (labels_flat: (R*Cw,) int32 device array, n_groups: int).
    """
    shift = encode.partition_shift(batch, cfg['g_h'], cfg['g_w'], cfg['shift_step'])
    labels, n_groups = encode.group_labels(R, Cw, cfg['g_h'], cfg['g_w'], *shift)
    return encode.flat_labels(labels), n_groups


def output_path(cfg):
    """.npz path under <scene sim dir>/results/, unique per
    configuration. Creates the directory if needed.
    """
    scene_cfg = cfg['scene_cfg']
    stem = (f"G{cfg['g_h']}x{cfg['g_w']}_B{cfg['batch_size']}"
            f"_shift{cfg['shift_step']}_{cfg['family']}_{cfg['sharing']}"
            f"_p{cfg['p_detect']}_{cfg['decode_mode']}")
    if cfg['hw_delay_probs']:
        probs_str = '-'.join(f'{v}x{p:.2f}' for v, p in sorted(cfg['hw_delay_probs'].items()))
        stem += f'_hwdelay{probs_str}'
        if cfg['hw_delay_mode'] != 'per_cycle':
            stem += f"_{cfg['hw_delay_mode']}"
    directory = f'{dc.scene_sim_dir(scene_cfg)}/results'
    os.makedirs(directory, exist_ok=True)
    return f"{directory}/{stem}{cfg['out_tag']}.npz"


def main():
    """Run one non-adaptive acquisition from start (or a checkpoint) to end.
    Writes a result .npz and periodic checkpoints; deletes the checkpoint
    on success. Ground truth is read only after the batch loop completes,
    so a VERIFY_BATCHES run returns before it's touched. Returns exit code
    0 on success.
    """
    cfg = configuration()
    scene_cfg = cfg['scene_cfg']
    R, Cw, RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T

    code_source = codes.CodeSource(
        scene_cfg, cfg['family'], sharing=cfg['sharing'])
    delay_source = (hw_delay.DelaySource(cfg['hw_delay_probs'])
                    if cfg['hw_delay_probs'] is not None else None)
    hw_delay_log = (f" {hw_delay.describe(cfg['hw_delay_probs'])} mode={cfg['hw_delay_mode']}"
                     if cfg['hw_delay_probs'] else '')
    print(f"scene={scene_cfg.name} G={cfg['g_h']}x{cfg['g_w']} B={cfg['batch_size']} "
          f"shift_step={cfg['shift_step']} {code_source.describe()} "
          f"p_detect={cfg['p_detect']} cycles={cfg['cycles']} "
          f"batches={cfg['n_batches']} decode={cfg['decode_mode']}{hw_delay_log}", flush=True)

    sampler = photon_sample.PhotonSampler(
        scene_cfg, cfg['batch_size'], cfg['p_detect'])
    accumulator = decode.new_accumulator(R, Cw, RES_T)
    raw_counts = decode.new_raw_counts(R, Cw, RES_T)

    # Depth-map trace: one (raw, reconstructed, bilateral) snapshot every
    # dc.CKPT_INTERVAL cycles, decoded as the accumulator stands at that cycle.
    assert cfg['cycles'] % dc.CKPT_INTERVAL == 0, \
        f"cycles={cfg['cycles']} must divide into dc.CKPT_INTERVAL={dc.CKPT_INTERVAL}-cycle snapshots"
    assert dc.CKPT_INTERVAL % cfg['batch_size'] == 0, \
        f"batch_size={cfg['batch_size']} must divide dc.CKPT_INTERVAL={dc.CKPT_INTERVAL}"
    n_ckpts = cfg['cycles'] // dc.CKPT_INTERVAL
    trace_raw = np.zeros((n_ckpts, R, Cw), dtype=np.float32)
    trace_recon = np.zeros((n_ckpts, R, Cw), dtype=np.float32)
    trace_bilat = np.zeros((n_ckpts, R, Cw), dtype=np.float32)
    trace_idx = 0

    stop_batch = (min(cfg['n_batches'], cfg['verify_batches'])
                  if cfg['verify_batches'] else cfg['n_batches'])
    if stop_batch != cfg['n_batches']:
        print(f'VERIFICATION MODE: stopping after batch {stop_batch}, '
              'ground truth will not be loaded', flush=True)

    result_path = output_path(cfg)
    checkpoint_path = checkpoint.path_for(result_path)
    acquisition = acquisition_config(cfg)
    total_readout_bits = 0
    start_batch = 0

    state = checkpoint.load(checkpoint_path, acquisition)
    if state is not None:
        start_batch = int(state['next_batch'])
        accumulator = cp.asarray(state['accumulator'])
        raw_counts = cp.asarray(state['raw_counts'])
        total_readout_bits = int(state['total_readout_bits'])
        sampler.accepted_by_category = state['accepted_by_category'].copy()
        sampler.rejected = int(state['rejected_photons'])
        trace_raw = state['trace_raw']
        trace_recon = state['trace_recon']
        trace_bilat = state['trace_bilat']
        trace_idx = int(state['trace_idx'])
        print(f'RESUMING from batch {start_batch}; replaying RNG streams',
              flush=True)
        reposition_started = time.time()
        replayed = checkpoint.reposition_streams(
            sampler, code_source, start_batch,
            labels_for_batch=None if cfg['sharing'] != 'shared' else
            (lambda b: shared_labels_for(cfg, R, Cw, b)))
        if delay_source is not None:
            if cfg['hw_delay_mode'] == 'per_batch_group':
                # n_groups only depends on the shift pattern, which is
                # constant across batches for SHIFT_STEP=0 (fixed block);
                # a sliding block would need this recomputed per batch.
                _, n_groups_ff = shared_labels_for(cfg, R, Cw, 0)
                delay_source.fast_forward_per_group(start_batch, n_groups_ff)
            else:
                delay_source.fast_forward(start_batch, R * Cw, cfg['batch_size'])
        print(f'  repositioned ({replayed} batches replayed, '
              f'{time.time() - reposition_started:.0f}s)', flush=True)

    labels_flat = None
    n_groups = None
    last_shift = None
    started = time.time()

    def write_checkpoint(next_batch):
        """Persist everything accumulated so far (next_batch: where a
        resumed run should start). Costs a ~1.2 GB device-to-host transfer,
        so this runs every CHECKPOINT_EVERY batches, not every batch.
        """
        checkpoint.save(
            checkpoint_path, next_batch, acquisition,
            dict(accumulator=cp.asnumpy(accumulator),
                 raw_counts=cp.asnumpy(raw_counts),
                 total_readout_bits=np.int64(total_readout_bits),
                 accepted_by_category=sampler.accepted_by_category,
                 rejected_photons=np.int64(sampler.rejected),
                 trace_raw=trace_raw, trace_recon=trace_recon,
                 trace_bilat=trace_bilat, trace_idx=np.int64(trace_idx)))

    # One iteration = one batch = B illumination cycles sharing a single
    # code and producing a single readout bit per group.
    for batch in range(start_batch, stop_batch):
        shift = encode.partition_shift(batch, cfg['g_h'], cfg['g_w'], cfg['shift_step'])

        # Rebuild the group map only when the lattice moved (once for a
        # fixed grid; most batches for a sliding one).
        if shift != last_shift:
            labels, n_groups = encode.group_labels(
                R, Cw, cfg['g_h'], cfg['g_w'], shift[0], shift[1])
            labels_flat = encode.flat_labels(labels)
            last_shift = shift

        code_pb = code_source.draw(batch, labels_flat, n_groups)
        arrivals = sampler.sample(batch)
        decode.accumulate_raw(raw_counts, arrivals, RES_T)

        if cfg['decode_mode'] == 'hier0':
            bits = encode.group_bits_hierarchical(code_pb, labels_flat, n_groups, arrivals)
        else:
            if delay_source is None:
                delay_bins = None
            elif cfg['hw_delay_mode'] == 'per_batch_group':
                delay_bins = delay_source.draw_per_group(labels_flat, n_groups, cfg['batch_size'])
            else:
                delay_bins = delay_source.draw(R * Cw, cfg['batch_size'])
            bits = encode.group_bits(code_pb, labels_flat, n_groups, arrivals,
                                      pixel_delay_bins=delay_bins)

        decode.backproject(accumulator, code_pb, bits, labels_flat)
        total_readout_bits += n_groups
        del code_pb, bits, arrivals  # the code alone is ~154 MB

        completed = batch + 1
        cycles_done = completed * cfg['batch_size']
        if cycles_done % dc.CKPT_INTERVAL == 0:
            maps = reconstruct.depth_maps(accumulator, raw_counts, scene_cfg, **cfg['bilateral'])
            trace_raw[trace_idx] = maps['raw']
            trace_recon[trace_idx] = maps['reconstructed']
            trace_bilat[trace_idx] = maps['bilateral']
            trace_idx += 1

        if completed % max(1, stop_batch // 20) == 0 or completed == stop_batch:
            print(f'  batch {completed}/{stop_batch}: groups={n_groups} '
                  f'bits={total_readout_bits} ({time.time() - started:.0f}s)', flush=True)
        if cfg['checkpoint_every'] and completed % cfg['checkpoint_every'] == 0 \
                and completed != stop_batch:
            write_checkpoint(completed)

    if stop_batch != cfg['n_batches']:
        write_checkpoint(stop_batch)
        print(f'VERIFIED through batch {stop_batch}; resumable checkpoint saved; '
              'ground truth was never loaded', flush=True)
        return 0

    maps = dict(raw=trace_raw, reconstructed=trace_recon, bilateral=trace_bilat)
    gt = reconstruct.load_ground_truth(scene_cfg)
    errors = {name: reconstruct.mae(trace[-1], gt) for name, trace in maps.items()}
    metadata = dict(
        scene=scene_cfg.name, group=[cfg['g_h'], cfg['g_w']],
        batch_size=cfg['batch_size'], shift_step=cfg['shift_step'],
        code=cfg['family'], code_sharing=cfg['sharing'],
        p_detect=cfg['p_detect'], cycles=cfg['cycles'],
        n_batches=cfg['n_batches'], decode=cfg['decode_mode'],
        total_readout_bits=total_readout_bits,
        bilateral_radius=cfg['bilateral']['radius'],
        bilateral_sigma_spatial=cfg['bilateral']['sigma_spatial'],
        bilateral_sigma_range=cfg['bilateral']['sigma_range'],
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
    print('Done.', flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        os.environ['B'] = sys.argv[1]
    sys.exit(main())
