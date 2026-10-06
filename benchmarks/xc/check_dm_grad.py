"""GPU-staged SCF on this card vs a stored FP64 reference (density, gradient).

Runs only the GPU part: StagedSCF(gpu_stages=(0, 1, 2), fp64_final=True,
GPU K, fp64_conv_tol 1e-7) -- the production configuration -- then
evaluates gradient / dipole / Mulliken charges from its converged orbitals
with the same stock code as ref_dm_grad.py and compares with the .npz.

Gates (exit 3 if one fails): |dE| < 1e-6 Ha (spec precision gate), max
|d grad| < 1e-5 Ha/bohr.  Density
metrics (max |dD|, relative Frobenius, dipole, Mulliken) are reported.

    $PY -B benchmarks/xc/check_dm_grad.py --ref benchmarks/results/ref_dm_grad/caffeine_def2-tzvp.npz
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..'))
import numpy as np  # noqa: E402

from dm_grad_common import host_info, log, observables, stock_mf  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402

GATE_E = 1e-6
GATE_GRAD = 1e-5


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ref', required=True)
    ap.add_argument('--conv-tol', type=float, default=1e-9)
    ap.add_argument('--no-gpu-k', dest='gpu_k', action='store_false')
    ap.add_argument('--out', default=None, help='JSON (default: next to --ref)')
    args = ap.parse_args(argv)
    ref = np.load(args.ref)
    meta = json.loads(str(ref['meta']))
    mol = build_mol(meta['system'], meta['basis'])
    log('START check_dm_grad %s/%s nao=%d vs %s (ref from %s %s) %s'
        % (meta['system'], meta['basis'], mol.nao, args.ref, meta['host'], meta['date'],
           host_info()))
    if not np.allclose(mol.atom_coords(), ref['atom_coords'], rtol=0, atol=1e-12):
        raise SystemExit('geometry differs from the reference file')

    from pyscf_wb97mv_fast.core.sgx_patch import apply as sgx_apply, revert as sgx_revert
    from pyscf_wb97mv_fast.staging.schedule import StagedSCF
    mf = stock_mf(mol)
    mf.conv_tol = args.conv_tol
    gpu_kwargs = {'k': True, 'k_tile_tol': 1e-11} if args.gpu_k else {}
    sgx_apply()
    try:
        staged = StagedSCF(mf, gpu_stages=(0, 1, 2), fp64_final=True,
                           gpu_kwargs=gpu_kwargs, fp64_conv_tol=1e-7)
        t = time.time()
        with staged:                 # as schedule.run_staged
            mf.kernel()
            staged.add_nonscf_vv10()
        e = float(mf.e_tot)
        t_scf = time.time() - t
    finally:
        sgx_revert()
    if not mf.converged:
        raise SystemExit('GPU-staged SCF did not converge')
    log('GPU-staged SCF e=%.10f cycles=%d wall=%.0fs (gpu_k=%s)'
        % (e, mf.cycles, t_scf, args.gpu_k))
    obs = observables(mol, mf.mo_coeff, mf.mo_occ, mf.mo_energy, e)
    dD = obs['dm'] - ref['dm']
    dg = obs['grad'] - ref['grad']
    res = dict(system=meta['system'], basis=meta['basis'], ref=args.ref, gpu_k=args.gpu_k,
               cycles=int(mf.cycles), t_scf=t_scf, t_grad=obs['t_grad'],
               dE=float(e - ref['e_tot']),
               dD_max=float(np.abs(dD).max()),
               dD_rel_fro=float(np.linalg.norm(dD) / np.linalg.norm(ref['dm'])),
               dgrad_max=float(np.abs(dg).max()),
               dgrad_rms=float(np.sqrt(np.mean(dg ** 2))),
               grad_max=float(np.abs(ref['grad']).max()),
               ddipole_D=float(np.linalg.norm(obs['dipole'] - ref['dipole'])),
               dcharge_max=float(np.abs(obs['charges'] - ref['charges']).max()),
               dgrad_per_atom=np.abs(dg).max(axis=1).tolist(), **host_info())
    res['gates'] = {'dE_1e-6': abs(res['dE']) < GATE_E,
                    'dgrad_1e-5': res['dgrad_max'] < GATE_GRAD,
                    'finite': bool(np.all(np.isfinite(obs['grad'])))}
    log('dE=%+.3e  max|dD|=%.2e  rel||dD||=%.2e  max|dg|=%.2e (rms %.2e, max|g| %.2e)  '
        '|d dipole|=%.2e D  max|dq|=%.2e'
        % (res['dE'], res['dD_max'], res['dD_rel_fro'], res['dgrad_max'], res['dgrad_rms'],
           res['grad_max'], res['ddipole_D'], res['dcharge_max']))
    ok = all(res['gates'].values())
    log('gates: %s -> %s' % (res['gates'], 'PASS' if ok else 'FAIL'))
    out = args.out or args.ref.replace('.npz', '_check_%s.json' % res['host'])
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    log('-> %s' % out)
    log('END check_dm_grad')
    return 0 if ok else 3


if __name__ == '__main__':
    sys.exit(main())
