"""One S2-settings get_veff on a real system: where the time goes, and the
VV10 FP32 error (inputs for the next optimization round).

    A  get_veff through install_gpu(overlap=True), second call (first warms
       the RawKernel / context): main-thread J/K vs worker GPU XC+VV10 vs
       time blocked in join -- is the GPU cycle K-bound or GPU-bound?
    C  VV10 alone: GPU FP32 gpu_nr_nlc_vxc vs the CPU FP64 nr_nlc_vxc on the
       same density and grid -- can the FP64 tail keep VV10 on the GPU?

Known values are not re-measured: the semilocal XC FP32 error (water27
-2.67e-5 Ha fixed-DM, benchmarks/results/fp32_attribution_water27.json) and
the CPU S2 cycle time (98-118 s).

S2 settings: XC grids level 3, VV10 grids level 3, SGX level 2, sgx_patch.

Usage (GPU idle, nothing else running):
    $PY -B benchmarks/xc/cycle_breakdown.py [--system water27]
Output: benchmarks/results/cycle_breakdown_<system>.json (+ stdout).
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
from pyscf import dft  # noqa: E402

from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.gpu import install  # noqa: E402
from pyscf_wb97mv_fast.gpu.vv10 import gpu_nr_nlc_vxc  # noqa: E402

XC = 'wb97m-v'
DM_DIR = os.path.join(os.path.dirname(HERE), 'experimental', 'cosx')


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', default='water27')
    args = ap.parse_args(argv)
    mol = build_mol(args.system, 'def2-svp')
    dm = np.load(os.path.join(DM_DIR, 'dm_%s_def2-svp.npy' % args.system))
    dm = (dm + dm.T) * 0.5
    out = {'system': args.system, 'nao': int(mol.nao)}

    sgx_patch.apply()
    try:
        mf = dft.RKS(mol, xc=XC).COSX(pjs=True)
        mf.grids.level = 3
        mf.nlcgrids.level = 3
        mf.with_df.grids_level_i = mf.with_df.grids_level_f = 2
        mf.build()
        hooks = install.install_gpu(mf, overlap=True)
        try:
            for call in (1, 2):
                mf._nsteps_direct = 0                  # full J/K both times
                t0 = time.perf_counter()
                mf.get_veff(mol, dm)
                wall = time.perf_counter() - t0
                tm = hooks.timings[-1]
                log('A get_veff #%d: wall %.1fs  jk %.1fs  gpu-xc %.1fs  wait %.1fs'
                    % (call, wall, tm['jk'], tm['xc'], tm['wait']))
            out['A_get_veff'] = dict(wall=wall, **tm)
            ctx_ni = mf._numint
        finally:
            hooks.restore_all()

        # C: VV10 FP32 (GPU) vs FP64 (CPU), same density and grid
        ctx = install.build_context(mol, backend='cupy', verify_ao=False)
        t0 = time.perf_counter()
        _, e32, v32 = gpu_nr_nlc_vxc(ctx, ctx_ni, mol, mf.nlcgrids, XC, dm)
        t_gpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, e64, v64 = ctx_ni.nr_nlc_vxc(mol, mf.nlcgrids, XC, dm)
        t_cpu = time.perf_counter() - t0
        out['C_vv10'] = {'e_fp64': float(e64), 'dE_fp32': float(e32 - e64),
                         'max_abs_dV': float(np.abs(v32 - v64).max()),
                         'npts_nlc': int(mf.nlcgrids.weights.size),
                         'gpu_s': t_gpu, 'cpu_s': t_cpu}
        log('C VV10 E_nl(FP64)=%.10f  dE(FP32-FP64)=% .3e  max|dV|=%.2e  gpu %.1fs  cpu %.1fs'
            % (e64, e32 - e64, out['C_vv10']['max_abs_dV'], t_gpu, t_cpu))
    finally:
        sgx_patch.revert()

    path = os.path.join(results_dir(), 'cycle_breakdown_%s.json' % args.system)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    log('-> %s' % path)


if __name__ == '__main__':
    main()
