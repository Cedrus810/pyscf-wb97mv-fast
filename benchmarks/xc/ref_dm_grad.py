"""FP64 CPU reference density matrix + gradient for one system, stored once.

Stock PySCF RKS(wb97m-v).COSX(pjs=True), no sgx_patch, no GPU, conv_tol
1e-9, the stock setting of every other reference here (the COSX energy
jitters ~5e-9 between FP64 cycles, so 1e-11 cannot converge -- bromazepam
failed at 50 cycles, 2026-10-06).  Output (.npz): e_tot, dm, grad, dipole, charges,
mo_coeff/mo_occ/mo_energy, atom_coords and a JSON meta string (settings,
host, wall times).  check_dm_grad.py compares any card's GPU-staged run
against this file -- the reference is not recomputed there.

    $PY -B benchmarks/xc/ref_dm_grad.py --system benchmarks/systems/drugs/caffeine.xyz \
        --basis def2-tzvp --out benchmarks/results/ref_dm_grad/caffeine_def2-tzvp.npz
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
import pyscf  # noqa: E402

from dm_grad_common import host_info, log, observables, stock_mf  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', required=True)
    ap.add_argument('--basis', default='def2-tzvp')
    ap.add_argument("--conv-tol", type=float, default=1e-9)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    mol = build_mol(args.system, args.basis)
    log('START ref_dm_grad %s/%s nao=%d natm=%d %s argv=%s'
        % (args.system, args.basis, mol.nao, mol.natm, host_info(), sys.argv[1:]))
    mf = stock_mf(mol)
    mf.conv_tol = args.conv_tol
    t = time.time()
    e = mf.kernel()
    t_scf = time.time() - t
    if not mf.converged:
        raise SystemExit('reference SCF did not converge -- nothing written')
    log('SCF e=%.10f cycles=%d wall=%.0fs' % (e, mf.cycles, t_scf))
    obs = observables(mol, mf.mo_coeff, mf.mo_occ, mf.mo_energy, e)
    if not np.all(np.isfinite(obs['grad'])):
        raise SystemExit('reference gradient not finite -- nothing written')
    log('gradient wall=%.0fs  max|g|=%.3e  dipole=%s D'
        % (obs['t_grad'], np.abs(obs['grad']).max(), np.round(obs['dipole'], 6)))
    meta = dict(system=args.system, basis=args.basis, xc='wb97m-v', cosx='pjs',
                conv_tol=args.conv_tol, cycles=int(mf.cycles), t_scf=t_scf,
                t_grad=obs['t_grad'], pyscf=pyscf.__version__,
                grids_level=int(mf.grids.level), nlcgrids_level=int(mf.nlcgrids.level),
                sgx_grids_level=[mf.with_df.grids_level_i, mf.with_df.grids_level_f],
                sgx_grid_response=False, grid_response=obs['grid_response'],
                date=time.strftime('%F %T'), **host_info())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, e_tot=e, dm=obs['dm'], grad=obs['grad'], dipole=obs['dipole'],
             charges=obs['charges'], mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ,
             mo_energy=mf.mo_energy, atom_coords=mol.atom_coords(),
             meta=json.dumps(meta))
    log('-> %s' % args.out)
    log('END ref_dm_grad')


if __name__ == '__main__':
    main()
