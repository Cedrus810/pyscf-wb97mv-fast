"""Compile every GPU kernel once on this card, before timing anything.

    python -m pyscf_wb97mv_fast.gpu.warmup

CuPy compiles our RawKernels / RawModule (and its own elementwise kernels)
for the device's architecture on first use and keeps the binaries in its
on-disk cache (~/.cupy/kernel_cache, or $CUPY_CACHE_DIR).  The first SCF on a
new card therefore pays ~145 s of compilation (4090: cycle 1 took 136.9 s
instead of 10.1 s, STATUS section 17); later runs hit the cache.  None of the
kernels is specialised by system size -- every (la, lb) class of the fused K
is compiled as one module -- so one small staged SCF through GPU XC, VV10 and
K compiles all of them.  Run once per architecture and cache directory.

Exit status 0 when the SCF converged with no GPU fallback, 1 otherwise (a
fallback means some kernel did not run, so the cache is not warm).
"""
import sys
import time
import warnings

FALLBACK_MARK = 'pyscf_wb97mv_fast GPU'      # gpu.install._warn_fallback
NO_GPU_MARK = 'GPU fast path not'            # gpu.install: CuPy/CUDA missing


def fallbacks(messages):
    """The warning texts that mean a GPU path did not run."""
    return [m for m in messages if FALLBACK_MARK in m or NO_GPU_MARK in m]


def warmup(basis='def2-svp'):
    """One staged SCF on the water dimer with GPU XC + VV10 + K.
    Returns (exit status, report lines)."""
    from pyscf_wb97mv_fast.core.testsystems import build_mol
    from pyscf_wb97mv_fast.staging.schedule import run_staged

    mol = build_mol('water_dimer', basis)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        info = run_staged(mol, gpu_stages=(0, 1, 2), fp64_final=True,
                          gpu_kwargs={'k': True})
    wall = time.perf_counter() - t0
    bad = fallbacks([str(w.message) for w in caught])
    n_gpu = sum(1 for h in info['history'] if h['gpu'] and not h.get('fp64'))
    report = ['water_dimer/%s: %.1f s, %d cycles (%d on the GPU FP32 path), '
              'converged=%s' % (basis, wall, info['cycles'], n_gpu,
                                info['converged'])]
    report += ['GPU fallback: %s' % m.splitlines()[0] for m in bad]
    ok = info['converged'] and n_gpu > 0 and not bad
    report.append('kernel cache warm' if ok else 'NOT warm: see above')
    return (0 if ok else 1), report


def main():
    rc, report = warmup()
    for line in report:
        print(line)
    return rc


if __name__ == '__main__':
    sys.exit(main())
