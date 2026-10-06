#!/usr/bin/env python
"""OpenBLAS-in-OpenMP probe: stock CPU get_veff timings under the current
BLAS thread settings (2026-10-03 performance notes, section 3).

Every stock log has ~30k "OpenBLAS Warning : Detect OpenMP Loop" lines: the
pthreads OpenBLAS is called from inside PySCF's OpenMP regions.  Run this
twice in a row and compare the get_veff times; if they differ, the stock
baseline (and so every speedup G) carries the effect too:

    benchmarks/run_logged.sh <log file> $PY -B benchmarks/blas_probe.py
    OPENBLAS_NUM_THREADS=1 benchmarks/run_logged.sh <log file> $PY -B benchmarks/blas_probe.py

The warnings go to stderr between the timestamped lines, so
`grep -c "Detect OpenMP"` per section shows where they come from.
Nothing is written besides stdout/stderr.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()

import numpy  # noqa: E402
from pyscf import dft, lib  # noqa: E402

from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402


def log(msg):
    sys.stderr.flush()
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', default='water27')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--repeat', type=int, default=3)
    args = ap.parse_args()

    log('OMP threads (pyscf lib)=%d  OPENBLAS_NUM_THREADS=%s  OMP_NUM_THREADS=%s'
        % (lib.num_threads(), os.environ.get('OPENBLAS_NUM_THREADS', 'unset'),
           os.environ.get('OMP_NUM_THREADS', 'unset')))

    a = numpy.random.default_rng(0).standard_normal((4000, 4000))
    a.dot(a)
    t = time.perf_counter()
    for _ in range(3):
        a.dot(a)
    log('section numpy dgemm 4000^3 x3: %.2f s' % (time.perf_counter() - t))

    sgx_patch.revert()                       # stock PySCF path, as profile_scf --stock
    mol = build_mol(args.system, args.basis)
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    mf.build()
    mf.with_df.build(level=mf.with_df.grids_level_f)   # the late-SCF SGX grid
    dm = mf.get_init_guess()
    log('%s nao=%d; init guess done' % (args.system, mol.nao))
    for i in range(args.repeat):
        mf._nsteps_direct = 0                # set by kernel/pre_kernel normally;
        mf._in_scf = False                   # 0 = full SGX build, no grid switch
        t = time.perf_counter()
        mf.get_veff(mol, dm)                 # no dm_last: one full J/K + XC + VV10
        log('section get_veff #%d%s: %.2f s'
            % (i + 1, ' (incl. grids build)' if i == 0 else '', time.perf_counter() - t))


if __name__ == '__main__':
    main()
