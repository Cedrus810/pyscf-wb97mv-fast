"""S5 Task 8: K performance gate -- GPU fused K vs CPU get_k_only.

Same machine, OMP_NUM_THREADS=16; for full (omega=0) and LR (omega=wB97M-V)
separately, time the CPU sgx_jk.get_k_only and the GPU GpuKBuilder path
(warm-up run first, second run timed) and print |dE_K| = 0.25|tr(D dK)|.

Acceptance (S5 plan Task 8): GPU time <= 0.2 x CPU for both full and LR
(target 0.1), with the Task 4 accuracy thresholds on |dE_K|.  On the
development card this script only checks correctness; the timing gate
runs on the dedicated timing node.

Usage:
    $PY -B benchmarks/sgx/bench_gpu_k.py --system water27 --sgx-level 2
Output: benchmarks/results/bench_gpu_k_<system>.json (+ stdout).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _paths import add_repo_to_syspath, results_dir  # noqa: E402

add_repo_to_syspath()
import cupy as cp  # noqa: E402
from pyscf import dft  # noqa: E402
from pyscf.sgx import sgx_jk  # noqa: E402

from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu  # noqa: E402

XC = 'wb97m-v'
DM_DIR = os.path.join(os.path.dirname(HERE), 'experimental', 'cosx')


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def load_dm(system, mol):
    path = os.path.join(DM_DIR, 'dm_%s_def2-svp.npy' % system)
    if os.path.exists(path):
        dm = np.load(path)
    else:
        mf = dft.RKS(mol, xc=XC)
        mf.conv_tol = 1e-8
        mf.kernel()
        dm = mf.make_rdm1()
    return (dm + dm.T) * 0.5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', default='water27')
    ap.add_argument('--sgx-level', type=int, default=2)
    ap.add_argument('--tile-tol', type=float, default=1e-11)
    ap.add_argument('--repeats', type=int, default=2,
                    help='GPU runs; the last one is timed')
    args = ap.parse_args()

    mol = build_mol(args.system, 'def2-svp')
    dm = load_dm(args.system, mol)
    mf = dft.RKS(mol, xc=XC).COSX(pjs=True)
    sgx = mf.with_df
    sgx.grids_level_i = sgx.grids_level_f = args.sgx_level
    sgx.build(level=args.sgx_level)
    omega_lr = float(mf._numint.rsh_and_hybrid_coeff(XC)[0])
    out = {'system': args.system, 'sgx_level': args.sgx_level,
           'nao': int(mol.nao),
           'ngrids_sgx': int(sgx.grids.weights.size),
           'tile_tol': args.tile_tol,
           'device': str(cp.cuda.runtime.getDeviceProperties(0)['name'])}
    log('%s nao=%d SGX L%d: %d grid points, omega_lr=%.2f, device=%s'
        % (args.system, mol.nao, args.sgx_level, out['ngrids_sgx'],
           omega_lr, out['device']))

    for label, omega in (('full', 0.0), ('lr', omega_lr)):
        rec = {'omega': omega}
        with mol.with_range_coulomb(omega):
            t0 = time.perf_counter()
            k_cpu = sgx_jk.get_k_only(sgx, dm, hermi=1)
            rec['cpu_s'] = time.perf_counter() - t0

            builder = GpuKBuilder(mol, cp, tile_tol=args.tile_tol)
            k_gpu = None
            for rep in range(args.repeats):
                t0 = time.perf_counter()
                k_gpu = get_k_only_gpu(builder, sgx, dm, hermi=1)
                rec['gpu_s'] = time.perf_counter() - t0
                rec['gpu_kernel_s'] = builder.last_stats['kernel']
            rec['kept_ratio'] = (builder.last_stats['pairs_kept']
                                 / builder.last_stats['pairs_total'])

        dE = float(0.25 * np.einsum('ij,ji->', dm, k_gpu - k_cpu))
        rec['dE_K_ha'] = dE
        rec['max_abs_dK'] = float(np.abs(k_gpu - k_cpu).max())
        rec['rel_fro_dK'] = float(np.linalg.norm(k_gpu - k_cpu)
                                  / np.linalg.norm(k_cpu))
        rec['speedup'] = rec['cpu_s'] / rec['gpu_s']
        out[label] = rec
        log('%-4s CPU %.1fs | GPU %.1fs (kernel %.1fs) -> %.1fx | kept %.1f%% | '
            'max|dK|=%.2e |dE_K|=%.2e Ha'
            % (label, rec['cpu_s'], rec['gpu_s'], rec['gpu_kernel_s'],
               rec['speedup'], 100 * rec['kept_ratio'],
               rec['max_abs_dK'], dE))
        gate = rec['gpu_s'] <= 0.2 * rec['cpu_s']
        out[label + '_gate_0.2x'] = bool(gate)
        log('%-4s gate GPU <= 0.2 x CPU: %s' % (label, 'PASS' if gate else 'FAIL'))

    path = os.path.join(results_dir(), 'bench_gpu_k_%s.json' % args.system)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    log('-> %s' % path)


if __name__ == '__main__':
    main()
