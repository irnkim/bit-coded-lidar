"""Adaptive quadtree partition from first-pass photons (no ground truth).

Inputs (per pixel, from the first pass):
  dA, dB : (R, C) int depth-bin estimates from the odd / even batch halves
  hist   : (R, C, T) first-pass counts, for the pooled-histogram flatness test
Levels, coarse to fine: 12x16 -> 6x8 -> 3x4.

Split signal (per block, per level):
  structure: S_b = 1.4826 * MAD over pixels p in b of dbar_p = (dA_p+dB_p)/2
  noise:     sigma_full = 1.4826 * MAD_p(dA_p - dB_p) / 2, over the whole scene
  split b if S_b > K_SPLIT * sigma_full

Flatness signal (per block):
  H_b(t) = sum_{p in b} hist_p(t);  z_b = (max_t H_b - median_t H_b) / sqrt(median_t H_b)
  merge four sibling blocks back into their parent if all four have z < Z_MIN
"""
import numpy as np

LEVELS = [(12, 16), (6, 8), (3, 4)]
K_SPLIT = 3.0
Z_MIN = 5.0


def block_reduce(a, gh, gw, fn):
    R, C = a.shape[:2]
    out = np.empty((R // gh, C // gw) + a.shape[2:], dtype=np.float64)
    for i in range(R // gh):
        for j in range(C // gw):
            out[i, j] = fn(a[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw])
    return out


def block_stats(dA, dB, hist, gh, gw):
    dbar = 0.5 * (dA.astype(np.float64) + dB.astype(np.float64))
    S = block_reduce(dbar, gh, gw, lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))
    Hb = block_reduce(hist.astype(np.float64), gh, gw, lambda x: x.sum(axis=(0, 1)))
    beta = np.median(Hb, axis=-1)
    peak = Hb.max(axis=-1)
    z = (peak - beta) / np.sqrt(np.maximum(beta, 1.0))
    return S, z


def sigma_full(dA, dB):
    delta = dA.astype(np.float64) - dB.astype(np.float64)
    return 1.4826 * np.median(np.abs(delta - np.median(delta))) / 2.0


def build_partition(dA, dB, hist):
    """List of leaves (r0, c0, gh, gw) that tile the scene."""
    R, C = dA.shape
    stats = {lvl: block_stats(dA, dB, hist, *lvl) for lvl in LEVELS}
    sig = sigma_full(dA, dB)
    leaves = []

    def visit(r0, c0, lvl_idx):
        gh, gw = LEVELS[lvl_idx]
        D, z = stats[LEVELS[lvl_idx]]
        i, j = r0 // gh, c0 // gw
        split = lvl_idx + 1 < len(LEVELS) and D[i, j] > K_SPLIT * sig
        if split:
            ch, cw = LEVELS[lvl_idx + 1]
            for dr in (0, gh // 2):
                for dc in (0, gw // 2):
                    visit(r0 + dr, c0 + dc, lvl_idx + 1)
        else:
            leaves.append((r0, c0, gh, gw))

    gh0, gw0 = LEVELS[0]
    for r0 in range(0, R, gh0):
        for c0 in range(0, C, gw0):
            visit(r0, c0, 0)

    # merge pass: four equal-sized sibling leaves that are all flat -> one parent leaf
    merged = True
    while merged:
        merged = False
        leaf_set = {(r, c, h, w) for (r, c, h, w) in leaves}
        for (r, c, h, w) in list(leaf_set):
            if (h, w) == LEVELS[0]:
                continue
            ph, pw = 2 * h, 2 * w
            pr, pc = r - (r % ph), c - (c % pw)
            sibs = [(pr, pc, h, w), (pr, pc + w, h, w), (pr + h, pc, h, w), (pr + h, pc + w, h, w)]
            if all(s in leaf_set for s in sibs):
                lvl_idx = LEVELS.index((h, w))
                _, z = stats[(h, w)]
                zs = [z[s[0] // h, s[1] // w] for s in sibs]
                if all(zz < Z_MIN for zz in zs):
                    for s in sibs:
                        leaves.remove(s)
                    leaves.append((pr, pc, ph, pw))
                    merged = True
                    break
    return leaves


def label_map(leaves, R, C):
    lab = np.full((R, C), -1, dtype=np.int32)
    for k, (r, c, h, w) in enumerate(leaves):
        lab[r:r + h, c:c + w] = k
    assert (lab >= 0).all(), 'leaves do not tile the scene'
    return lab


if __name__ == '__main__':
    # synthetic check: smooth wall on the left half, an edge on the right, strong peaks everywhere
    rng = np.random.default_rng(0)
    R, C, T = 48, 64, 200
    truth = np.full((R, C), 60)
    truth[:, 32:] = np.where(np.arange(C - 32)[None, :] % 4 < 2, 60, 120)  # fine stripes on the right
    hist = np.zeros((R, C, T))
    for r in range(R):
        for c in range(C):
            hist[r, c, truth[r, c]] = 400
    hist += rng.poisson(1.0, size=hist.shape)
    dA = truth + rng.integers(-1, 2, size=truth.shape)   # half-decodes, noisy
    dB = truth + rng.integers(-1, 2, size=truth.shape)
    leaves = build_partition(dA, dB, hist)
    sizes = {}
    for (_, _, h, w) in leaves:
        sizes[(h, w)] = sizes.get((h, w), 0) + 1
    print('leaf sizes:', sizes)
    print('left-half leaf sizes:', sorted({(h, w) for (r, c, h, w) in leaves if c < 32}))
    print('right-half leaf sizes:', sorted({(h, w) for (r, c, h, w) in leaves if c >= 32}))
    lab = label_map(leaves, R, C)
    print('tiles scene:', (lab >= 0).all())
