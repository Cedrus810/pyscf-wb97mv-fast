#!/usr/bin/env python
"""G2a gate: B-path decomposition (+ optional experimental SR bounds) vs stock
A path. One K build per kernel and setting, bounds pre-built outside the timed
region (as in tolerance_scan.py), SGX work counted per kernel via the shim.

    python benchmarks/experimental/cosx/bench_sr_screening.py water27
    python benchmarks/experimental/cosx/bench_sr_screening.py water64 --erfc-bounds
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))  # benchmarks/, for _paths
import _paths  # noqa: E402
_paths.add_repo_to_syspath()
sys.path.insert(0, HERE)  # sibling sgx_locality.py

from pyscf import dft, lib  # noqa: E402

from sgx_locality import Shim, get_dm  # noqa: E402

from pyscf_wb97mv_fast.core import sgx_patch  # noqa: E402
from pyscf_wb97mv_fast.experimental.cosx.sr_screening import ErfcSRBounds  # noqa: E402
from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402

DIRECT_SCF_TOL = 1e-13
# G2a thresholds (P2a)
RN_MAX = 0.5
BA_MAX = 0.85


def make_objects(mol, etol, tol_full, tol_sr, use_bpath):
    """Build the SGX objects a setting needs, with bounds pre-built."""
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    omega, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    base = mf.with_df
    base.build()

    def make(om, tol):
        r = base.copy()
        r._rsh_df = {}
        r._vjopt = None
        r._overlap_correction_matrix = None
        r.sgx_tol_energy = tol
        r.sgx_tol_potential = 'auto'
        with r.mol.with_range_coulomb(om):
            r.build()
        return r

    objs = dict(full=make(0.0, etol if not use_bpath else tol_full))
    if use_bpath:
        objs['att'] = make(-omega, tol_sr)
    else:
        objs['att'] = make(omega, etol)
    return objs, (omega, alpha, hyb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('--basis', default='def2-svp')
    ap.add_argument('--etol', type=float, default=1e-8)
    ap.add_argument('--tol-full', type=float, default=1e-11)
    ap.add_argument('--erfc-bounds', action='store_true',
                    help='enable the experimental SR position bounds (G2a)')
    args = ap.parse_args()
    tag = os.path.splitext(os.path.basename(args.system))[0] + '_' + args.basis
    mol = build_mol(args.system, args.basis)
    dm = get_dm(mol, tag)
    sgx_patch.apply()
    shim = Shim(mol)
    if args.erfc_bounds:
        ErfcSRBounds(slack_bohr=4.0, enable=True).attach()

    def k_of(r, om):
        with r.mol.with_range_coulomb(om):
            return r.get_jk(dm, 1, None, with_j=False, with_k=True,
                            direct_scf_tol=DIRECT_SCF_TOL)[1]

    def counted(r, om):
        """One K build of one kernel with the shim counting only that build."""
        shim.start(record=False)
        t0 = time.perf_counter()
        k = k_of(r, om)
        wall = time.perf_counter() - t0
        cnt, _ = shim.stop()
        return k, dict(tasks=cnt['calls'], ints=cnt['ints'],
                       t_int_thread_s=cnt['t_int_thread_s'], wall_s=wall)

    # tight reference for the accuracy of every setting (A path, stock 'auto')
    ref_objs, (omega, alpha, hyb) = make_objects(mol, 'auto', 'auto', 'auto', False)
    k_ref = (hyb * k_of(ref_objs['full'], 0.0)
             + (alpha - hyb) * k_of(ref_objs['att'], omega))

    out = {}
    for name, use_bpath, etol, tol_full, tol_sr in (
            ('A_path', False, args.etol, args.etol, args.etol),
            ('B_path', True, args.etol, args.etol, args.etol),
            ('B_path+budget', True, args.etol, args.tol_full, args.etol)):
        objs, _ = make_objects(mol, etol, tol_full, tol_sr, use_bpath)
        om_att = -omega if use_bpath else omega
        k_of(objs['full'], 0.0)                      # warm-up: builds the screening bounds
        k_of(objs['att'], om_att)
        k_full, c_full = counted(objs['full'], 0.0)
        k_att, c_att = counted(objs['att'], om_att)
        if use_bpath:
            k = alpha * k_full - (alpha - hyb) * k_att
        else:
            k = hyb * k_full + (alpha - hyb) * k_att
        out[name] = dict(full=c_full, att=c_att,
                         ints=c_full['ints'] + c_att['ints'],
                         wall_s=c_full['wall_s'] + c_att['wall_s'],
                         dK_vs_ref=float(abs(k - k_ref).max()),
                         dE_vs_ref=float(0.25 * abs((dm * (k - k_ref).T).sum())))
        print(name, json.dumps(out[name]), flush=True)

    # G2a (P2a) on the accuracy-matched B setting:
    #   R_N  = SR ints (B+budget) / LR ints (A)
    #   B/A  = total K wall (B+budget) / total K wall (A)
    b = out['B_path+budget']
    r_n = b['att']['ints'] / max(out['A_path']['att']['ints'], 1)
    ba_wall = b['wall_s'] / out['A_path']['wall_s']
    gate = dict(R_N=r_n, BA_wall=ba_wall,
                dK_A=out['A_path']['dK_vs_ref'], dK_B=b['dK_vs_ref'],
                passed=bool(r_n <= RN_MAX and ba_wall <= BA_MAX
                            and b['dK_vs_ref'] <= max(10 * out['A_path']['dK_vs_ref'], 1e-10)))
    out['info'] = dict(system=args.system, basis=args.basis, nao=mol.nao,
                       threads=lib.num_threads(), etol=args.etol,
                       tol_full=args.tol_full, erfc_bounds=args.erfc_bounds)
    out['gate'] = gate
    path = os.path.join(_paths.results_dir(), f'bench_sr_screening_{tag}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f"G2a (B_path+budget): R_N = SR/LR ints = {r_n:.3f} (<= {RN_MAX}), "
          f"B/A K wall = {ba_wall:.3f} (<= {BA_MAX}), "
          f"dK_B = {b['dK_vs_ref']:.1e} vs dK_A = {out['A_path']['dK_vs_ref']:.1e} "
          f"-> {gate['passed']}")
    print('wrote', path)


if __name__ == '__main__':
    main()
