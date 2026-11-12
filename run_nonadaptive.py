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
    TRIG_K              harmonic count (frequency, F) for CODE=trig only,
                        default 3 -- ignored by every other family
    CYCLES              illumination cycles, default dataset_config.M
    BILAT_RADIUS        default 5
    BILAT_SIGMA_SPATIAL default 2.0
    BILAT_SIGMA_RANGE   default 0.15
    VERIFY_BATCHES      stop after this many batches without evaluating, for
                        smoke tests -- ground truth is never loaded
    OUT_TAG             suffix appended to the output filename
    SAVE_BITS           1 to also save every batch's readout bits plus a
                        per-checkpoint raw-brightness guide image, to
                        <result>.bits.npz -- for replaying the same
                        measurement sequence under a different filter
                        without redrawing photons. Off by default;
                        requires shift_step=0 (n_groups must stay constant
                        across batches to fit one fixed-width array).
    LOAD_BITS           1 to SKIP the acquisition loop entirely and load
                        an already-saved <result>.bits.npz (from a prior
                        SAVE_BITS=1 run of this exact config) instead --
                        for trying a new decode (CUM_BILATERAL, ICM_SWEEPS)
                        against already-measured data without redrawing
                        photons OR codes. CODE/CODE_MODE/K_CODES/TRIG_K are
                        read from the bits file's own saved metadata in
                        this mode (ignored from the environment), so the
                        replay is guaranteed to match whatever produced it.
    CODE_MODE           continuous (default): codes.CodeSource draws a
                        fresh code every BATCH, held fixed for all B
                        cycles in it (the only mode before this option
                        existed).
                        kcodebook: codes.KCodebookSource predefines K_CODES
                        distinct codes ONCE for the whole run (shared
                        across every pixel, not per-pixel); each pixel,
                        each CYCLE (not batch), independently redraws a
                        uniformly random index into that fixed codebook.
                        Requires CODE_SHARING=unshared.
    K_CODES             codebook size for CODE_MODE=kcodebook; required
                        (>0) in that mode, ignored otherwise.
    CUM_BILATERAL       1 to also replay the batch's readout bits (either
                        just-collected, if SAVE_BITS=1, or LOAD_BITS=1's
                        loaded ones) through BOTH an independent and a
                        cumulative (recursive) bilateral filter at every
                        checkpoint, saving a (cycles, mae_indep,
                        mae_cumulative, delta_pct) trace to
                        <result>_cumulative_vs_independent.npz.
                        Independent: filters a COPY of the accumulator
                        every checkpoint -- the stored accumulator itself
                        is never touched, so every checkpoint starts fresh
                        from pure raw evidence (this is the SAME
                        'bilateral' map depth_maps() already computes, not
                        a new filter). Cumulative: filters the accumulator
                        and REPLACES it in place, so smoothing compounds
                        into every later checkpoint. Requires SAVE_BITS=1
                        or LOAD_BITS=1.
    ICM_SWEEPS          >0 to also run that many iterative-decode (1-bit
                        compressed sensing) sweeps: each pixel's true
                        histogram is modeled as a single spike at an
                        unknown bin, found by coordinate descent (Jacobi
                        sweeps) that directly minimizes sign-mismatches
                        against the observed group bits, instead of the
                        adjoint/backproject decoder. Reports three
                        variants per sweep -- ICM-recon (raw ICM argmax),
                        ICM-bilat (independent bilateral filter of ICM's
                        own evidence, never fed back), ICM-cum-bilat
                        (cumulative bilateral filter FED BACK into the
                        next sweep's coordinate descent) -- saved to
                        <result>_iterative_decode.npz. Requires
                        SAVE_BITS=1 or LOAD_BITS=1, and CODE_MODE=continuous
                        (kcodebook not yet supported here -- each cycle's
                        code differs within a batch in that mode, which
                        ICM's per-batch scoring doesn't yet account for).

Backend (GPU by default, CPU as a fallback when no GPU is reachable) is
chosen by BACKEND=gpu|cpu, read by xp_backend.py -- see that module's
docstring. CPU mode works but is NOT optimized for raw speed (no BLAS
tricks for code generation, etc.); use it when a GPU just isn't available,
not as a performance choice.

Ground truth is read only after the full batch loop finishes, inside
reconstruct.py.  A run stopped early by VERIFY_BATCHES returns before that.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xp_backend import cp, cupyx
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

    # Harmonic count (frequency, F) for the trig/cosine-wave family only;
    # ignored (but still carried in the config/fingerprint) for every other
    # family, so it never needs special-casing at the call sites below.
    trig_k = int(os.environ.get('TRIG_K', codes.TRIG_K_HARM))

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

    code_mode = os.environ.get('CODE_MODE', 'continuous')
    if code_mode not in ('continuous', 'kcodebook'):
        raise ValueError(f'CODE_MODE must be continuous or kcodebook, got {code_mode!r}')
    k_codes = int(os.environ.get('K_CODES', 0))
    if code_mode == 'kcodebook':
        if k_codes <= 0:
            raise ValueError('CODE_MODE=kcodebook requires K_CODES > 0')
        if sharing != 'unshared':
            raise ValueError('CODE_MODE=kcodebook only supports CODE_SHARING=unshared')
        if hw_delay_probs_raw:
            raise ValueError('CODE_MODE=kcodebook does not support HW_DELAY_PROBS')
        if decode_mode != 'majority0':
            raise ValueError('CODE_MODE=kcodebook only supports DECODE_MODE=majority0')

    save_bits = bool(int(os.environ.get('SAVE_BITS', 0)))
    load_bits = bool(int(os.environ.get('LOAD_BITS', 0)))
    cum_bilateral = bool(int(os.environ.get('CUM_BILATERAL', 0)))
    icm_sweeps = int(os.environ.get('ICM_SWEEPS', 0))
    if (cum_bilateral or icm_sweeps > 0) and not (save_bits or load_bits):
        raise ValueError('CUM_BILATERAL/ICM_SWEEPS require SAVE_BITS=1 or LOAD_BITS=1')
    if icm_sweeps > 0 and code_mode == 'kcodebook':
        raise NotImplementedError(
            'ICM_SWEEPS does not yet support CODE_MODE=kcodebook (see module docstring)')
    if save_bits and shift_step != 0:
        raise ValueError('SAVE_BITS=1 requires shift_step=0 (constant n_groups)')

    return dict(
        scene_cfg=scene_cfg, g_h=g_h, g_w=g_w, shift_step=shift_step,
        family=family, sharing=sharing, p_detect=p_detect, trig_k=trig_k,
        code_mode=code_mode, k_codes=k_codes,
        batch_size=batch_size, cycles=cycles, decode_mode=decode_mode,
        hw_delay_probs=hw_delay_probs, hw_delay_mode=hw_delay_mode,
        n_batches=cycles // batch_size,
        verify_batches=int(os.environ.get('VERIFY_BATCHES', 0)),
        checkpoint_every=int(os.environ.get('CHECKPOINT_EVERY', 500)),
        out_tag=os.environ.get('OUT_TAG', ''),
        save_bits=save_bits, load_bits=load_bits,
        cum_bilateral=cum_bilateral, icm_sweeps=icm_sweeps,
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

    trig_k is included only when family == 'trig', where it actually
    changes the draw -- including it unconditionally for every family
    would shift every OTHER family's fingerprint too (trig_k is read from
    TRIG_K regardless of family) and invalidate their in-progress
    checkpoints for no reason, the same bug HW_DELAY_MODE caused earlier.
    """
    extra = dict(trig_k=cfg['trig_k']) if cfg['family'] == 'trig' else {}
    if cfg['code_mode'] == 'kcodebook':
        extra['code_mode'] = cfg['code_mode']
        extra['k_codes'] = cfg['k_codes']
    return dict(
        scene=cfg['scene_cfg'].name, g_h=cfg['g_h'], g_w=cfg['g_w'],
        batch_size=cfg['batch_size'], shift_step=cfg['shift_step'],
        code=cfg['family'], sharing=cfg['sharing'], p_detect=cfg['p_detect'],
        cycles=cfg['cycles'], decode=cfg['decode_mode'],
        hw_delay_probs=cfg['hw_delay_probs'], hw_delay_mode=cfg['hw_delay_mode'],
        **extra,
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
    if cfg['family'] == 'trig' and cfg['trig_k'] != codes.TRIG_K_HARM:
        stem += f"_trigF{cfg['trig_k']}"
    if cfg['code_mode'] == 'kcodebook':
        stem += f"_kcodebook{cfg['k_codes']}"
    if cfg['hw_delay_probs']:
        probs_str = '-'.join(f'{v}x{p:.2f}' for v, p in sorted(cfg['hw_delay_probs'].items()))
        stem += f'_hwdelay{probs_str}'
        if cfg['hw_delay_mode'] != 'per_cycle':
            stem += f"_{cfg['hw_delay_mode']}"
    directory = f'{dc.scene_sim_dir(scene_cfg)}/results'
    os.makedirs(directory, exist_ok=True)
    return f"{directory}/{stem}{cfg['out_tag']}.npz"


def bits_output_path(cfg):
    """Sibling path for SAVE_BITS=1's readout-bits/guide-trace npz."""
    return output_path(cfg).removesuffix('.npz') + '.bits.npz'


def _fresh_code_source(cfg):
    """A new, from-batch-0 code source matching cfg -- CodeSource for
    CODE_MODE=continuous, KCodebookSource for kcodebook. Call this once
    per full replay pass (ICM needs one per sweep); a source is a
    stateful, non-seekable stream, so "fresh" always means batch 0.
    """
    if cfg['code_mode'] == 'kcodebook':
        return codes.KCodebookSource(cfg['scene_cfg'], cfg['k_codes'], family=cfg['family'],
                                      trig_k=cfg['trig_k'])
    return codes.CodeSource(cfg['scene_cfg'], cfg['family'], sharing=cfg['sharing'],
                             trig_k=cfg['trig_k'])


def _backproject_batch(cfg, source, batch, bits, labels_flat, accumulator):
    """Draw this batch's code from `source` and backproject `bits` through
    it into `accumulator`, in place -- the one step CUM_BILATERAL/ICM
    replay shares with the main acquisition loop, dispatched on
    CODE_MODE the same way the main loop is."""
    if cfg['code_mode'] == 'kcodebook':
        idx = source.draw(cfg['batch_size'])
        decode.backproject_kcodebook(accumulator, source.codebook, idx, bits, labels_flat)
    else:
        code_pb = source.draw(batch)
        decode.backproject(accumulator, code_pb, bits, labels_flat)


def run_cum_bilateral(cfg, scene_cfg, bits_trace, guide_trace, ckpt_interval,
                       labels_flat, n_groups, gt, result_path):
    """CUM_BILATERAL=1: replay bits_trace once, building an independent
    and a cumulative accumulator side by side (see module docstring), and
    save the per-checkpoint (indep, cumulative) MAE trace."""
    R, Cw, RES_T = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T
    per_bin_cm = scene_cfg.PER_BIN_CM
    n_batches = bits_trace.shape[0]
    B = cfg['batch_size']

    source = _fresh_code_source(cfg)
    accumulator_indep = decode.new_accumulator(R, Cw, RES_T)
    accumulator_cum = decode.new_accumulator(R, Cw, RES_T)

    print(f'CUM_BILATERAL: replaying {n_batches} batches ({n_batches * B} cycles, '
          f'{n_batches * B // ckpt_interval} checkpoints)', flush=True)
    started = time.time()
    ckpt_idx = 0
    rows = []
    for batch in range(n_batches):
        bits = cp.asarray(bits_trace[batch])
        _backproject_batch(cfg, source, batch, bits, labels_flat, accumulator_indep)
        _backproject_batch(cfg, source, batch, bits, labels_flat, accumulator_cum)

        cycles_done = (batch + 1) * B
        if cycles_done % ckpt_interval == 0:
            totals = cp.asarray(guide_trace[ckpt_idx], dtype=cp.float64)
            peak = float(totals.max())
            guide = (totals / peak if peak > 0 else totals).reshape(R, Cw)
            ckpt_idx += 1

            bins_indep = reconstruct.cross_bilateral_2d(
                accumulator_indep.reshape(R, Cw, RES_T), guide, 0, R)
            m_indep = reconstruct.mae(reconstruct.bins_to_cm(cp.asnumpy(bins_indep), per_bin_cm), gt)

            filtered_hist = reconstruct.cross_bilateral_2d_histogram(
                accumulator_cum.reshape(R, Cw, RES_T), guide, 0, R)
            bins_cum = filtered_hist.argmax(axis=-1)
            m_cum = reconstruct.mae(reconstruct.bins_to_cm(cp.asnumpy(bins_cum), per_bin_cm), gt)
            accumulator_cum[:] = filtered_hist.reshape(R * Cw, RES_T)

            delta = (m_indep - m_cum) / m_indep * 100 if m_indep else 0.0
            rows.append((cycles_done, m_indep, m_cum, delta))
            if ckpt_idx % max(1, (n_batches * B // ckpt_interval) // 20) == 0:
                print(f'  ckpt {ckpt_idx} (cycle {cycles_done}): indep={m_indep:.4f} '
                      f'cumulative={m_cum:.4f} delta={delta:+.1f}% '
                      f'({time.time() - started:.0f}s)', flush=True)

    rows = np.array(rows, dtype=np.float64)
    out_path = result_path.removesuffix('.npz') + '_cumulative_vs_independent.npz'
    np.savez(out_path, cycles=rows[:, 0], mae_indep=rows[:, 1], mae_cumulative=rows[:, 2],
             delta_pct=rows[:, 3])
    print(f'CUM_BILATERAL final: indep={rows[-1,1]:.4f} cumulative={rows[-1,2]:.4f} cm -- '
          f'wrote {out_path}', flush=True)


def run_icm(cfg, scene_cfg, bits_trace, guide_trace, labels_flat, n_groups, gt,
            init_bin, result_path):
    """ICM_SWEEPS=N: N rounds of Jacobi coordinate descent minimizing
    sign-mismatches against bits_trace directly (1-bit compressed
    sensing), instead of the adjoint/backproject decoder -- see module
    docstring for ICM-recon / ICM-bilat / ICM-cum-bilat. CODE_MODE=
    continuous only (validated in configuration())."""
    R, Cw, RES_T, per_bin_cm = scene_cfg.R, scene_cfg.Cw, scene_cfg.RES_T, scene_cfg.PER_BIN_CM
    n_pixels = R * Cw
    n_batches = bits_trace.shape[0]
    n_sweeps = cfg['icm_sweeps']

    totals = cp.asarray(guide_trace[-1], dtype=cp.float64)
    peak = float(totals.max())
    guide = (totals / peak if peak > 0 else totals).reshape(R, Cw)

    pixel_idx = cp.arange(n_pixels, dtype=cp.int64)
    bits_trace_gpu = cp.asarray(bits_trace, dtype=cp.int32)
    t_recon = cp.asarray(init_bin.reshape(n_pixels).astype(np.int32))
    t_cumbilat = cp.asarray(init_bin.reshape(n_pixels).astype(np.int32))

    init_mae = reconstruct.mae(reconstruct.bins_to_cm(init_bin, per_bin_cm), gt)
    print(f'ICM_SWEEPS: n_sweeps={n_sweeps} n_batches={n_batches} -- '
          f'init (backproject+argmax) MAE={init_mae:.4f}cm', flush=True)

    rows = [(0, init_mae, init_mae, init_mae)]
    recon_bin_np = cumbilat_bin_np = init_bin
    for sweep in range(1, n_sweeps + 1):
        started = time.time()
        source = _fresh_code_source(cfg)
        match_recon = cp.zeros((n_pixels, RES_T), dtype=cp.int32)
        match_cumbilat = cp.zeros((n_pixels, RES_T), dtype=cp.int32)

        for batch in range(n_batches):
            code_pb = source.draw(batch)
            y_b = bits_trace_gpu[batch][labels_flat]
            mask = (y_b != 0)[:, None]
            code_f = code_pb.astype(cp.float32)
            y_f = y_b[:, None].astype(cp.float32)

            for t_state, match_acc in ((t_recon, match_recon), (t_cumbilat, match_cumbilat)):
                old_contribution = code_pb[pixel_idx, t_state].astype(cp.int32)
                group_sum = cp.zeros(n_groups, dtype=cp.int32)
                cupyx.scatter_add(group_sum, labels_flat, old_contribution)
                leave_one_out = (group_sum[labels_flat] - old_contribution).astype(cp.float32)
                predicted = cp.sign(leave_one_out[:, None] + code_f)
                match_acc += ((predicted == y_f) & mask).astype(cp.int32)
            del code_pb, code_f, y_b, y_f, mask

        t_recon = match_recon.argmax(axis=-1).astype(cp.int32)
        recon_bin_np = cp.asnumpy(t_recon).reshape(R, Cw)
        mae_recon = reconstruct.mae(reconstruct.bins_to_cm(recon_bin_np, per_bin_cm), gt)

        hist_recon = match_recon.astype(cp.float32).reshape(R, Cw, RES_T)
        bilat_bins = reconstruct.cross_bilateral_2d(hist_recon, guide, 0, R)
        mae_bilat = reconstruct.mae(reconstruct.bins_to_cm(cp.asnumpy(bilat_bins), per_bin_cm), gt)

        hist_cumbilat = match_cumbilat.astype(cp.float32).reshape(R, Cw, RES_T)
        filtered_hist = reconstruct.cross_bilateral_2d_histogram(hist_cumbilat, guide, 0, R)
        cumbilat_bin_np = cp.asnumpy(filtered_hist.argmax(axis=-1))
        mae_cumbilat = reconstruct.mae(reconstruct.bins_to_cm(cumbilat_bin_np, per_bin_cm), gt)
        t_cumbilat = cp.asarray(cumbilat_bin_np.reshape(n_pixels).astype(np.int32))
        del match_recon, match_cumbilat, hist_recon, hist_cumbilat, filtered_hist, bilat_bins

        rows.append((sweep, mae_recon, mae_bilat, mae_cumbilat))
        print(f'  sweep {sweep}/{n_sweeps}: ICM-recon={mae_recon:.4f} ICM-bilat={mae_bilat:.4f} '
              f'ICM-cum-bilat={mae_cumbilat:.4f} cm ({time.time() - started:.0f}s)', flush=True)

    rows = np.array(rows, dtype=np.float64)
    out_path = result_path.removesuffix('.npz') + '_iterative_decode.npz'
    np.savez(out_path, sweeps=rows[:, 0], mae_icm_recon=rows[:, 1], mae_icm_bilat=rows[:, 2],
              mae_icm_cumbilat=rows[:, 3], init_mae=init_mae,
              depth_icm_recon=recon_bin_np.astype(np.float32),
              depth_icm_cumbilat=cumbilat_bin_np.astype(np.float32), n_sweeps=n_sweeps)
    print(f'ICM final: ICM-recon={rows[-1,1]:.4f} ICM-bilat={rows[-1,2]:.4f} '
          f'ICM-cum-bilat={rows[-1,3]:.4f} cm (init {init_mae:.4f}cm) -- wrote {out_path}', flush=True)


def _run_post_decode(cfg, scene_cfg, result_path, bits_trace, guide_trace, ckpt_interval,
                      gt, init_bin):
    """Shared tail for CUM_BILATERAL/ICM_SWEEPS, called from both the
    normal acquisition path and the LOAD_BITS fast path."""
    if not (cfg['cum_bilateral'] or cfg['icm_sweeps'] > 0):
        return
    R, Cw = scene_cfg.R, scene_cfg.Cw
    labels, n_groups = encode.group_labels(R, Cw, cfg['g_h'], cfg['g_w'], 0, 0)
    labels_flat = encode.flat_labels(labels)
    if cfg['cum_bilateral']:
        run_cum_bilateral(cfg, scene_cfg, bits_trace, guide_trace, ckpt_interval,
                           labels_flat, n_groups, gt, result_path)
    if cfg['icm_sweeps'] > 0:
        run_icm(cfg, scene_cfg, bits_trace, guide_trace, labels_flat, n_groups, gt,
                init_bin, result_path)


def main_load_bits(cfg, scene_cfg, result_path):
    """LOAD_BITS=1: skip the acquisition loop entirely, load a prior
    SAVE_BITS=1 run's bits/guide trace plus its result's gt/recon map, and
    jump straight to CUM_BILATERAL/ICM_SWEEPS. Returns exit code 0."""
    bits_path = bits_output_path(cfg)
    if not os.path.exists(bits_path):
        raise FileNotFoundError(
            f'LOAD_BITS=1 but {bits_path} does not exist -- run this exact config with '
            'SAVE_BITS=1 first')
    if not os.path.exists(result_path):
        raise FileNotFoundError(
            f'LOAD_BITS=1 but {result_path} does not exist -- run this exact config with '
            'SAVE_BITS=1 first')

    d = np.load(bits_path, allow_pickle=True)
    bits_trace = d['bits']
    guide_trace = d['guide_trace']
    ckpt_interval = int(d['ckpt_interval'])
    print(f'LOAD_BITS: loaded {bits_path} ({bits_trace.shape[0]} batches, '
          f'{bits_trace.shape[1]} groups)', flush=True)

    r = np.load(result_path, allow_pickle=True)
    gt = r['gt'].astype(np.float64)
    init_cm = r['depth_reconstructed'][-1]
    init_bin = np.clip(np.round(init_cm / scene_cfg.PER_BIN_CM - 0.5), 0, scene_cfg.RES_T - 1)

    _run_post_decode(cfg, scene_cfg, result_path, bits_trace, guide_trace, ckpt_interval,
                      gt, init_bin)
    print('Done.', flush=True)
    return 0


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
    result_path = output_path(cfg)

    if cfg['load_bits']:
        return main_load_bits(cfg, scene_cfg, result_path)

    if cfg['code_mode'] == 'kcodebook':
        code_source = codes.KCodebookSource(scene_cfg, cfg['k_codes'], family=cfg['family'],
                                             trig_k=cfg['trig_k'])
    else:
        code_source = codes.CodeSource(
            scene_cfg, cfg['family'], sharing=cfg['sharing'], trig_k=cfg['trig_k'])
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

    if cfg['save_bits']:
        _, n_groups_fixed = encode.group_labels(R, Cw, cfg['g_h'], cfg['g_w'], 0, 0)
        bits_trace = np.zeros((cfg['n_batches'], n_groups_fixed), dtype=np.float32)
        guide_trace = np.zeros((n_ckpts, R * Cw), dtype=np.float32)

    stop_batch = (min(cfg['n_batches'], cfg['verify_batches'])
                  if cfg['verify_batches'] else cfg['n_batches'])
    if stop_batch != cfg['n_batches']:
        print(f'VERIFICATION MODE: stopping after batch {stop_batch}, '
              'ground truth will not be loaded', flush=True)

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
        if cfg['save_bits']:
            bits_trace = state['bits_trace']
            guide_trace = state['guide_trace']
        print(f'RESUMING from batch {start_batch}; replaying RNG streams',
              flush=True)
        reposition_started = time.time()
        if cfg['code_mode'] == 'kcodebook':
            sampler.skip_to(start_batch)
            code_source.fast_forward(start_batch, cfg['batch_size'])
            replayed = start_batch
        else:
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
        arrays = dict(accumulator=cp.asnumpy(accumulator),
                      raw_counts=cp.asnumpy(raw_counts),
                      total_readout_bits=np.int64(total_readout_bits),
                      accepted_by_category=sampler.accepted_by_category,
                      rejected_photons=np.int64(sampler.rejected),
                      trace_raw=trace_raw, trace_recon=trace_recon,
                      trace_bilat=trace_bilat, trace_idx=np.int64(trace_idx))
        if cfg['save_bits']:
            arrays['bits_trace'] = bits_trace
            arrays['guide_trace'] = guide_trace
        checkpoint.save(checkpoint_path, next_batch, acquisition, arrays)

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

        arrivals = sampler.sample(batch)
        decode.accumulate_raw(raw_counts, arrivals, RES_T)

        if cfg['code_mode'] == 'kcodebook':
            idx = code_source.draw(cfg['batch_size'])
            bits = encode.group_bits_kcodebook(code_source.codebook, idx, labels_flat,
                                                n_groups, arrivals)
            decode.backproject_kcodebook(accumulator, code_source.codebook, idx, bits, labels_flat)
            del idx
            code_pb = None
        else:
            code_pb = code_source.draw(batch, labels_flat, n_groups)
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
        if cfg['save_bits']:
            bits_trace[batch] = cp.asnumpy(bits)
        del code_pb, bits, arrivals  # the code alone is ~154 MB

        completed = batch + 1
        cycles_done = completed * cfg['batch_size']
        if cycles_done % dc.CKPT_INTERVAL == 0:
            maps = reconstruct.depth_maps(accumulator, raw_counts, scene_cfg, **cfg['bilateral'])
            trace_raw[trace_idx] = maps['raw']
            trace_recon[trace_idx] = maps['reconstructed']
            trace_bilat[trace_idx] = maps['bilateral']
            if cfg['save_bits']:
                guide_trace[trace_idx] = cp.asnumpy(
                    raw_counts.reshape(R * Cw, RES_T).sum(axis=-1).astype(cp.float32))
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
        code_mode=cfg['code_mode'], k_codes=cfg['k_codes'],
    )
    path = reconstruct.save(result_path, maps, gt, errors, metadata)
    checkpoint.discard(checkpoint_path)

    if cfg['save_bits']:
        bits_out = bits_output_path(cfg)
        replay_note = (
            "bits[i] is batch i's encode group_bits output; guide_trace[j] is the raw "
            'photon-count brightness image (flat R*Cw) at checkpoint j '
            '((j+1)*dc.CKPT_INTERVAL cycles); codes are regenerated by replaying, from '
            'batch 0 (not seekable), ' +
            (f'codes.KCodebookSource(scene_cfg, k_codes, family, trig_k).draw(batch_size) '
             'per batch for the per-cycle idx (codebook itself fixed at construction)'
             if cfg['code_mode'] == 'kcodebook' else
             'codes.CodeSource(scene_cfg, family, sharing, trig_k).draw(batch_index)') +
            ' -- identical to this run.')
        np.savez(bits_out, bits=bits_trace, guide_trace=guide_trace,
                 ckpt_interval=dc.CKPT_INTERVAL, n_groups=bits_trace.shape[1],
                 g_h=cfg['g_h'], g_w=cfg['g_w'], batch_size=cfg['batch_size'],
                 trig_k=cfg['trig_k'], code=cfg['family'],
                 code_mode=cfg['code_mode'], k_codes=cfg['k_codes'],
                 note=replay_note)
        print(f'  saved bits -> {bits_out}', flush=True)

    print('MAE (cm): raw / recon / bilat = '
          f"{errors['raw']:.2f} / {errors['reconstructed']:.2f} / "
          f"{errors['bilateral']:.2f}", flush=True)
    print(f'wrote {path}', flush=True)

    if cfg['save_bits']:
        init_bin = np.clip(np.round(trace_recon[-1] / scene_cfg.PER_BIN_CM - 0.5),
                            0, RES_T - 1)
        _run_post_decode(cfg, scene_cfg, result_path, bits_trace, guide_trace,
                          dc.CKPT_INTERVAL, gt, init_bin)

    print('Done.', flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        os.environ['B'] = sys.argv[1]
    sys.exit(main())
