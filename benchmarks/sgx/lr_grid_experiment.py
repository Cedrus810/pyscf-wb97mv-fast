"""S5 Task 0 Step 1: independent LR grid experiment (CPU, spec section 8.5).

Question: should the long-range K (erf(omega r)/r, omega = 0.3 for
wB97M-V) keep its own, coarser SGX grid instead of sharing the full-range
grid?  For each (full level, LR level) configuration of water27:

* LR grid error   ||K_LR - K_LR,exact||_F / ||K_LR,exact||  and the energy
  form 0.25 * tr(D dK)  (Ha) against the exact (non-COSX) LR K,
* wall time of one LR K build,
* |dE| of the complete staged SCF against the known reference
  -2061.0026652808 Ha (never recomputed).

The exact K is parsed once and cached to
benchmarks/results/k_exact_<system>_<omega>.npy; later runs only read it.

Conclusion rule (spec 8.5, goes into benchmarks/sgx/FINDINGS.md): LR error
still far below 1e-4 Ha and the cost down to ~0.5x -> an independent LR
grid is worthwhile (Task 7 uses it); a clearly larger error -> an
LR-specific correction instead of an independent grid.

Usage (CPU node, OMP_NUM_THREADS=16):
    $PY -B benchmarks/sgx/lr_grid_experiment.py --system water27 \
        --e-ref -2061.0026652808 --omega 0.3
Output: benchmarks/results/lr_grid_<system>.json (+ stdout).
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
from pyscf import dft, scf  # noqa: E402
from pyscf.sgx import sgx as sgx_mod  # noqa: E402

from pyscf_wb97mv_fast.core.sgx_patch import apply as sgx_apply  # noqa: E402
from pyscf_wb97mv_fast.core.sgx_patch import get_jk as patched_get_jk  # noqa: E402
from pyscf_wb97mv_fast.core.sgx_patch import revert as sgx_revert  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.staging.schedule import StagedSCF  # noqa: E402

XC = 'wb97m-v'
E_REF_DEFAULT = -2061.0026652808
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


def get_exact_k(system, mol, dm, omega):
    """Exact (non-COSX) exchange K, parsed once then cached."""
    path = os.path.join(results_dir(),
                        'k_exact_%s_%s.npy' % (system, '%.6f' % omega))
    if os.path.exists(path):
        return np.load(path), True
    log('computing the exact K (omega=%g) -- one-time parse' % omega)
    hf = scf.hf.RHF(mol)
    t0 = time.perf_counter()
    k = np.asarray(hf.get_k(mol, dm, hermi=1, omega=omega or None))
    np.save(path, k)
    log('exact K done in %.1fs -> %s' % (time.perf_counter() - t0, path))
    return k, False


def apply_lr_level(level):
    """Wrap core.sgx_patch's get_jk so every LR copy carries its own grid at
    `level`.  Call AFTER sgx_apply(); undone by sgx_revert().  The marker
    detects the full-range df.build recursion (which resets the copy's grid
    to the full level) and rebuilds the copy at `level`."""
    def get_jk(self, dm, hermi=1, vhfopt=None, with_j=True, with_k=True,
               direct_scf_tol=1e-13, omega=None):
        if omega is None:
            return patched_get_jk(self, dm, hermi, vhfopt, with_j, with_k,
                                  direct_scf_tol)
        key = '%.6f' % omega
        rsh_df = self._rsh_df.get(key)
        if rsh_df is None:
            rsh_df = self.copy()
            rsh_df._rsh_df = None
            rsh_df._vjopt = None
            rsh_df._overlap_correction_matrix = None
            rsh_df._pjs_data = None
            self._rsh_df[key] = rsh_df
        marker = (id(self.grids), self.grids.weights.size)
        if (getattr(rsh_df, '_s5_lr_marker', None) != marker
                or rsh_df.grids is None):
            rsh_df.grids_level_i = rsh_df.grids_level_f = level
            rsh_df.build(level=level)
            rsh_df._s5_lr_marker = marker
        with rsh_df.mol.with_range_coulomb(omega):
            return sgx_mod.SGX._stock_get_jk(
                rsh_df, dm, hermi, vhfopt, with_j, with_k, direct_scf_tol)
    sgx_mod.SGX.get_jk = get_jk


def mark_stock():
    """Preserve the stock get_jk before sgx_patch replaces it, so the LR
    branch can call it without recursion through the patches."""
    if '_stock_get_jk' not in sgx_mod.SGX.__dict__:
        sgx_mod.SGX._stock_get_jk = sgx_mod.SGX.get_jk


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--system', default='water27')
    ap.add_argument('--omega', type=float, default=0.3)
    ap.add_argument('--e-ref', type=float, default=E_REF_DEFAULT)
    ap.add_argument('--sgx-level', type=int, default=2)
    ap.add_argument('--lr-level', type=int, default=1)
    ap.add_argument('--no-scf', action='store_true',
                    help='skip the staged SCFs (K builds only)')
    args = ap.parse_args()
    system = args.system
    omega = args.omega

    mol = build_mol(system, 'def2-svp')
    dm = load_dm(system, mol)
    k_exact, cached = get_exact_k(system, mol, dm, omega)
    out = {'system': system, 'omega': omega, 'e_ref': args.e_ref,
           'sgx_level': args.sgx_level, 'lr_level': args.lr_level,
           'exact_k': 'cached' if cached else 'computed'}
    log('%s nao=%d, exact LR K %s' % (system, mol.nao, out['exact_k']))

    mark_stock()
    configs = [(args.sgx_level, args.sgx_level),
               (args.sgx_level, args.lr_level),
               (args.lr_level, args.lr_level)]
    for full_lv, lr_lv in configs:
        tag = 'full L%d / LR L%d' % (full_lv, lr_lv)
        log('--- %s ---' % tag)
        rec = {'full_level': full_lv, 'lr_level': lr_lv}
        sgx_apply()
        try:
            if lr_lv != full_lv:
                apply_lr_level(lr_lv)
            m1 = build_mol(system, 'def2-svp')
            mf = dft.RKS(m1, xc=XC).COSX(pjs=True)
            sgx = mf.with_df
            sgx.grids_level_i = sgx.grids_level_f = full_lv
            sgx.build(level=full_lv)
            t0 = time.perf_counter()
            k_lr = np.asarray(mf.get_k(m1, dm, hermi=1, omega=omega))
            t_lr = time.perf_counter() - t0
            rec['lr_k_s'] = t_lr
            rec['lr_k_err_fro'] = float(np.linalg.norm(k_lr - k_exact)
                                        / np.linalg.norm(k_exact))
            rec['lr_k_err_max'] = float(np.abs(k_lr - k_exact).max())
            rec['lr_dE_ha'] = float(0.25 * np.einsum('ij,ji->', dm,
                                                     k_lr - k_exact))
            log('%s: LR K %.1fs, ||dK||/||K||=%.2e, 0.25 tr(D dK)=%+.3e Ha'
                % (tag, t_lr, rec['lr_k_err_fro'], rec['lr_dE_ha']))

            if not args.no_scf:
                m2 = build_mol(system, 'def2-svp')
                mf2 = dft.RKS(m2, xc=XC).COSX(pjs=True)
                mf2.conv_tol = 1e-9
                with StagedSCF(mf2):
                    mf2.kernel()
                rec['scf_e'] = float(mf2.e_tot)
                rec['scf_converged'] = bool(mf2.converged)
                rec['scf_dE_vs_ref'] = float(mf2.e_tot - args.e_ref)
                log('%s: staged SCF e=%.10f dE_vs_ref=%+.3e conv=%s'
                    % (tag, mf2.e_tot, rec['scf_dE_vs_ref'], mf2.converged))
        finally:
            sgx_revert()
        out['L%d_L%d' % (full_lv, lr_lv)] = rec

    path = os.path.join(results_dir(), 'lr_grid_%s.json' % system)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    log('-> %s' % path)
    log('CONCLUSION RULE: LR error << 1e-4 Ha at ~0.5x cost -> independent '
        'LR grid worthwhile (record in benchmarks/sgx/FINDINGS.md, use in '
        'Task 7); clearly larger error -> LR-specific correction instead.')


if __name__ == '__main__':
    main()
