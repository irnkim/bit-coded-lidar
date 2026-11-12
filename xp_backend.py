"""Backend selection: GPU (cupy) by default, CPU (numpy) when BACKEND=cpu.

Every module in this pipeline that used to do `import cupy as cp` /
`import cupyx` does `from xp_backend import cp, cupyx` instead -- nothing
else about their code changes (cp.zeros, cp.asarray, cp.random.default_rng,
cp.asnumpy, cupyx.scatter_add all keep working); this module alone decides
what `cp`/`cupyx` actually point at.

BACKEND defaults to 'gpu' so every existing invocation (the dispatcher,
already-saved checkpoints, every script written before this existed) is
completely unaffected unless BACKEND=cpu is set explicitly.

CPU mode intentionally does not try to be fast on its own -- the earlier
pure-numpy ports in compressiveLidar/code (run_b_sweep_cpu.py etc.) carry
the perf tricks (precomputed trig basis + BLAS matmul for code generation,
cached per-tile CDFs for photon sampling) that made CPU-only runs
practical; this shim exists so run_nonadaptive.py works at all without a
GPU, not so it's equally fast.
"""
import os

BACKEND = os.environ.get('BACKEND', 'gpu').lower()
if BACKEND not in ('gpu', 'cpu'):
    raise ValueError(f'BACKEND must be gpu or cpu, got {BACKEND!r}')

if BACKEND == 'gpu':
    import cupy as cp
    import cupyx
else:
    import numpy as cp

    if not hasattr(cp, 'asnumpy'):
        cp.asnumpy = lambda x: x  # numpy arrays already live on host

    class _NoOpMemPool:
        def free_all_blocks(self):
            pass

    if not hasattr(cp, 'get_default_memory_pool'):
        cp.get_default_memory_pool = lambda: _NoOpMemPool()

    class _CupyxShim:
        @staticmethod
        def scatter_add(dest, indices, values):
            """Same semantics as cupyx.scatter_add: dest[indices[i]] += values[i]
            for every i, with repeats accumulating rather than overwriting
            (unlike plain fancy-index assignment) -- numpy's np.add.at is the
            exact CPU equivalent."""
            cp.add.at(dest, indices, values)

    cupyx = _CupyxShim()
