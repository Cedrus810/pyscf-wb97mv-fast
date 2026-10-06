"""Option (b): FP32 SCF + one FP64 energy evaluation at the final density.

benchmarks/xc/FINDINGS.md section 6 (b).  The FP32 XC error grows with the
system (water27 ~2.7e-5 Ha at fixed density), so FP32 iterations alone can
never meet the spec's 1e-6 layer-2 budget.  The SCF energy is variational:
evaluating the FP64 energy functional at the FP32-converged density should
leave only a second-order error.  This script measures that, per system:

    ref      stock PySCF, CPU FP64, full SCF
    fp32     the same SCF through install_gpu (default FP32 XC + VV10 flows)
    final64  hooks removed, mf.energy_tot(dm_fp32) on the CPU in FP64

All three share the settings: RKS(wB97M-V).COSX(pjs=True), core.sgx_patch
applied, conv_tol, default grids.  Criterion (spec section 8 layer 2):
|E_final64 - E_ref| <= 1e-6 Ha.  Also reported: |E_fp32 - E_ref|, cycles,
and the number of GPU->CPU fallback warnings (must be 0, or the fp32 run
did not actually use the GPU path).

Known references are reused, never recomputed: --e-ref SYSTEM=ENERGY skips
the ref SCF for that system (e.g. water27=-2061.0026652808, the stock CPU FP64
value in benchmarks/results/scf_gate_water27_def2-svp.json).

Usage (GPU idle, nothing else running):
    $PY -B benchmarks/xc/fp32_final_fp64.py [--systems water_dimer water27 chain30]
        [--e-ref water27=-2061.0026652808 ...]
Output: benchmarks/results/fp32_final_fp64_<system>.json after EACH system
(+ stdout), so a partial run keeps what finished.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _paths import add_repo_to_syspath, results_dir  # noqa: E402

add_repo_to_syspath()
from pyscf import dft  # noqa: E402

from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.gpu import install  # noqa: E402

XC = 'wb97m-v'
GATE = 1e-6


def make_mf(mol, conv_tol):
    mf = dft.RKS(mol, xc=XC).COSX(pjs=True)
    mf.conv_tol = conv_tol
    return mf


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    sys.stdout.flush()


def run(system, conv_tol, e_ref_known=None):
    mol = build_mol(system, 'def2-svp')
    log('%s: nao=%d natm=%d' % (system, mol.nao, mol.natm))

    ref = None
    if e_ref_known is None:
        t0 = time.perf_counter()
        ref = make_mf(mol, conv_tol)
        e_ref = ref.kernel()
        t_ref = time.perf_counter() - t0
        log('  ref     e=%.10f conv=%s cycles=%d wall=%.0fs'
            % (e_ref, ref.converged, ref.cycles, t_ref))
    else:
        e_ref, t_ref = e_ref_known, None
        log('  ref     e=%.10f (given, not recomputed)' % e_ref)

    t0 = time.perf_counter()
    mf = make_mf(mol, conv_tol)
    hooks = install.install_gpu(mf)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            e_fp32 = mf.kernel()
        finally:
            hooks.restore_all()
    fallbacks = [str(w.message).splitlines()[0] for w in caught
                 if 'falling back' in str(w.message)]
    t_fp32 = time.perf_counter() - t0
    dm = mf.make_rdm1()
    log('  fp32    e=%.10f conv=%s cycles=%d wall=%.0fs fallbacks=%d'
        % (e_fp32, mf.converged, mf.cycles, t_fp32, len(fallbacks)))

    # hooks are gone: this is the stock CPU FP64 energy functional
    ni = mf._numint
    for name in ('nr_rks', 'nr_nlc_vxc'):
        bound = getattr(ni, name)
        assert getattr(bound, '__func__', None) is getattr(type(ni), name), \
            '%s still hooked after restore_all()' % name
    t0 = time.perf_counter()
    e_final = float(mf.energy_tot(dm=dm))
    t_final = time.perf_counter() - t0
    log('  final64 e=%.10f wall=%.0fs' % (e_final, t_final))

    out = {
        'system': system, 'basis': 'def2-svp', 'xc': XC, 'nao': int(mol.nao),
        'natm': int(mol.natm), 'conv_tol': conv_tol, 'gate': GATE,
        'e_ref': float(e_ref), 'e_fp32': float(e_fp32), 'e_final64': e_final,
        'dE_fp32': float(e_fp32 - e_ref), 'dE_final64': float(e_final - e_ref),
        'pass_final64': bool(abs(e_final - e_ref) <= GATE),
        'e_ref_source': 'computed' if ref is not None else 'given',
        'converged': {'ref': None if ref is None else bool(ref.converged),
                      'fp32': bool(mf.converged)},
        'cycles': {'ref': None if ref is None else int(ref.cycles), 'fp32': int(mf.cycles)},
        'gpu_fallback_warnings': len(fallbacks), 'fallback_samples': fallbacks[:5],
        'dm_maxdiff_vs_ref': (None if ref is None
                              else float(np.abs(dm - ref.make_rdm1()).max())),
        'wall_s': {'ref': t_ref, 'fp32': t_fp32, 'final64': t_final},
    }
    path = os.path.join(results_dir(), 'fp32_final_fp64_%s.json' % system)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    log('  dE_fp32=% .3e  dE_final64=% .3e  (gate %.0e: %s)  dPmax=%s  -> %s'
        % (out['dE_fp32'], out['dE_final64'], GATE,
           'PASS' if out['pass_final64'] else 'FAIL', out['dm_maxdiff_vs_ref'], path))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--systems', nargs='+', default=['water_dimer', 'water27', 'chain30'])
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--e-ref', action='append', default=[], metavar='SYSTEM=ENERGY',
                    help='known reference energy; skips that ref SCF')
    args = ap.parse_args(argv)
    known = {k: float(v) for k, v in (item.split('=', 1) for item in args.e_ref)}
    sgx_patch.apply()
    try:
        for system in args.systems:
            run(system, args.conv_tol, known.get(system))
    finally:
        sgx_patch.revert()
    log('all done')


if __name__ == '__main__':
    main()
