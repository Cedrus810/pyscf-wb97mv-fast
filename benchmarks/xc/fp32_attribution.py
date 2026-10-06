"""Where does the FP32 XC energy error come from?  (spec sections 6, 8, 10)

Context: the FP32 XC flow misses the layer-2 budget (water dimer SCF
|dE| ~ 2.6e-6 Ha > 1e-6; ledger section 3).  ``ctx.dtype`` lifts coords,
basis parameters, AO evaluation AND the density matrix together, so the
dtype x gemm_dtype grid alone cannot separate the sources.  This script
injects one FP32 rounding source at a time into an otherwise FP64 pipeline
and reports the XC energy shift vs the FP64 flow.

Fixed density matrix, same grid, same active-shell screening and the same
density formulas as gpu.xc.gpu_nr_rks.  The SCF energy is variational, so
the SCF-level error is ~ this first-order fixed-density shift (dimer:
3.0e-6 fixed-DM vs 2.6e-6 SCF) -- the table predicts which lift closes
the budget before any SCF is run.

Variants (AO = values + gradients, DM = density matrix, GEMM = the
contraction to rho / grad rho / tau):

    ref        AO FP64 eval,          DM FP64, GEMM FP64  (the FP64 flow)
    prod       AO FP32 eval,          DM FP32, GEMM FP32  (install_gpu default)
    lift_gemm  AO FP32 eval,          DM FP32, GEMM FP64  (dtype=f32, gemm_dtype=f64)
    ao_eval    AO FP32 eval,          DM FP64, GEMM FP64  (FP32 coords + FP32 AO arithmetic)
    ao_store   AO FP64 eval -> FP32,  DM FP64, GEMM FP64  (AO storage rounding only)
    dm_store   AO FP64,               DM FP32, GEMM FP64  (DM rounding only)
    gemm       AO FP64 -> FP32,       DM FP32, GEMM FP32  (rounded inputs + FP32 arithmetic)

ao_eval split further -- FP64 AO arithmetic, one evaluator input rounded to
FP32 (and back), DM FP64, GEMM FP64:

    in_exps    primitive exponents
    in_coeffs  contraction coefficients
    in_c2s     cartesian -> spherical matrices
    in_coords  grid coordinates and shell centers (relative to the origin)
    in_all     all four (ao_eval - in_all = the FP32 AO arithmetic itself)

Component swaps (the prod value of ONE density component put into ref):
    swap_rho, swap_grad, swap_tau

Per-point breakdown of prod - ref: by log10(rho), by the iso-orbital
indicator z = tau_W / tau (tau - tau_W cancels as z -> 1), and by the
element / distance of the nearest nucleus.

Usage (GPU idle):
    $PY -B benchmarks/xc/fp32_attribution.py [--systems water_dimer water27 chain30]
Output: benchmarks/results/fp32_attribution_<system>.json (+ stdout).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _paths import add_repo_to_syspath, results_dir  # noqa: E402

add_repo_to_syspath()
from pyscf import dft  # noqa: E402

from pyscf_wb97mv_fast.core.testsystems import build_mol  # noqa: E402
from pyscf_wb97mv_fast.gpu import device  # noqa: E402
from pyscf_wb97mv_fast.gpu.ao_eval import plan_blocks  # noqa: E402
from pyscf_wb97mv_fast.gpu.install import build_context  # noqa: E402
from pyscf_wb97mv_fast.gpu.xc import PyscfLibxcFunctional, gpu_nr_rks  # noqa: E402

XC = 'wb97m-v'
DM_DIR = os.path.join(os.path.dirname(HERE), 'experimental', 'cosx')
F32, F64 = 'float32', 'float64'
INPUTS = {'in_exps': ('exps',), 'in_coeffs': ('coeffs',), 'in_c2s': ('c2s',),
          'in_coords': ('coords',), 'in_all': ('exps', 'coeffs', 'c2s', 'coords')}
VARIANTS = ('ref', 'prod', 'lift_gemm', 'ao_eval', 'ao_store', 'dm_store',
            'gemm') + tuple(INPUTS)
SWAPS = {'swap_rho': slice(0, 1), 'swap_grad': slice(1, 4), 'swap_tau': slice(4, 5)}


def density(xp, ao, dm, gd):
    """rho (5, P) FP64 array; same formulas and cast points as gpu_nr_rks."""
    ao = ao if ao.dtype == gd else ao.astype(gd)
    dm = dm if dm.dtype == gd else dm.astype(gd)
    tmp = ao[0] @ dm
    rho = xp.empty((5, ao.shape[1]), dtype=xp.float64)
    rho[0] = (tmp * ao[0]).sum(axis=1)
    rho[1:4] = 2.0 * xp.einsum('pm,gpm->gp', tmp, ao[1:4])
    c1 = ao[1:4] @ dm
    rho[4] = 0.5 * (c1 * ao[1:4]).sum(axis=2).sum(axis=0)
    return rho


def _r32(xp, a):
    return xp.asarray(np.asarray(a, dtype=np.float32), dtype=np.float64)


def rounded_input_ctx(mol, backend, blksize, which):
    """FP64 evaluator whose `which` inputs are rounded to FP32 and back."""
    ctx = build_context(mol, backend=backend, blksize=blksize, verify_ao=False,
                        dtype=F64, gemm_dtype=F64)
    pack, xp = ctx.evaluator.pack, ctx.xp
    for g in pack.groups:
        if 'exps' in which:
            g.exps_d = _r32(xp, g.exps)
        if 'coeffs' in which:
            g.coeffs_d = _r32(xp, g.coeffs)
        if 'coords' in which:
            g.centers_d = _r32(xp, g.centers)
    if 'c2s' in which:
        pack.c2s = [None if c is None else np.asarray(c, dtype=np.float32).astype(np.float64)
                    for c in pack.c2s]
    return ctx


def load_dm(system, mol):
    path = os.path.join(DM_DIR, 'dm_%s_def2-svp.npy' % system)
    if os.path.exists(path):
        dm, source = np.load(path), path
    else:
        # no stored density: converge a plain (exact-K) wB97M-V SCF
        mf = dft.RKS(mol, xc=XC)
        mf.conv_tol = 1e-10
        mf.kernel()
        if not mf.converged:
            raise RuntimeError('reference SCF for %s did not converge' % system)
        dm, source = mf.make_rdm1(), 'RKS(%s) conv_tol=1e-10, e_tot=%.10f' % (XC, mf.e_tot)
    if dm.shape != (mol.nao, mol.nao):
        raise ValueError('%s: dm shape %s != nao %d' % (path, dm.shape, mol.nao))
    dm = (dm + dm.T) * 0.5
    ne = float(np.einsum('ij,ji->', dm, mol.intor('int1e_ovlp')))
    if abs(ne - mol.nelectron) > 1e-6:
        raise ValueError('%s: tr(DS)=%.8f != nelectron %d' % (source, ne, mol.nelectron))
    return dm, source


def nearest_atom(coords, atom_coords, chunk=65536):
    idx = np.empty(coords.shape[0], dtype=np.int64)
    dist = np.empty(coords.shape[0])
    for p0 in range(0, coords.shape[0], chunk):
        d = np.linalg.norm(coords[p0:p0 + chunk, None, :] - atom_coords[None], axis=2)
        idx[p0:p0 + chunk] = d.argmin(axis=1)
        dist[p0:p0 + chunk] = d.min(axis=1)
    return idx, dist


def binned(d, keys, labels):
    """Sum / sum|.| / count of the per-point shift d for each bin label."""
    out = {}
    for k, lab in enumerate(labels):
        m = keys == k
        out[lab] = {'sum': math.fsum(d[m]), 'abs_sum': math.fsum(np.abs(d[m])),
                    'npts': int(m.sum())}
    return out


def run(system, backend, blksize):
    t0 = time.perf_counter()
    mol = build_mol(system, 'def2-svp')
    dm, dm_source = load_dm(system, mol)
    mf = dft.RKS(mol, xc=XC)
    grids = mf.grids
    grids.build(with_non0tab=True)
    ni = mf._numint

    ctx64 = build_context(mol, backend=backend, blksize=blksize, verify_ao=False,
                          dtype=F64, gemm_dtype=F64)
    ctx32 = build_context(mol, backend=backend, blksize=blksize, verify_ao=False,
                          dtype=F32, gemm_dtype=F32)
    ctx_in = {k: rounded_input_ctx(mol, backend, blksize, w) for k, w in INPUTS.items()}
    xp = ctx64.xp
    ao_loc = ctx64.evaluator.pack.ao_loc
    coords_rel = grids.coords - ctx64.origin
    coords_r32 = coords_rel.astype(np.float32).astype(np.float64)
    npts = grids.coords.shape[0]
    rho = {k: np.empty((5, npts)) for k in VARIANTS}

    for i0, i1, shells in plan_blocks(mol, grids, blksize):
        idx = device.ao_indices(ao_loc, shells)
        dm64 = xp.asarray(device.gather_dm(dm, idx), dtype=xp.float64)
        dm32 = dm64.astype(F32)
        ao64 = ctx64.evaluator.eval(xp.asarray(coords_rel[i0:i1], dtype=F64), shells, 1)
        ao32 = ctx32.evaluator.eval(xp.asarray(coords_rel[i0:i1], dtype=F64), shells, 1)  # as gpu_nr_rks
        ao64r = ao64.astype(F32).astype(F64)
        dm64r = dm32.astype(F64)
        todo = {'ref': (ao64, dm64, F64),
                'prod': (ao32, dm32, F32),
                'lift_gemm': (ao32, dm32, F64),
                'ao_eval': (ao32, dm64, F64),
                'ao_store': (ao64r, dm64, F64),
                'dm_store': (ao64, dm64r, F64),
                'gemm': (ao64.astype(F32), dm32, F32)}
        for k, (a, d, gd) in todo.items():
            rho[k][:, i0:i1] = device.to_host(density(xp, a, d, gd))
        del ao32, ao64r, todo
        for k, which in INPUTS.items():
            c = coords_r32 if 'coords' in which else coords_rel
            a = ctx_in[k].evaluator.eval(xp.asarray(c[i0:i1], dtype=F64), shells, 1)
            rho[k][:, i0:i1] = device.to_host(density(xp, a, dm64, F64))
            del a
    t_density = time.perf_counter() - t0

    for k, sl in SWAPS.items():
        r = rho['ref'].copy()
        r[sl] = rho['prod'][sl]
        rho[k] = r

    func = PyscfLibxcFunctional(XC, ni)
    w = grids.weights
    e_pt, E, nelec = {}, {}, {}
    for k, r in rho.items():
        exc = np.asarray(func.eval(r)[0])
        e_pt[k] = w * r[0] * exc
        E[k] = math.fsum(e_pt[k])
        nelec[k] = math.fsum(w * r[0])

    # the emulation must reproduce the real flows
    n_cpu, e_cpu, _ = ni.nr_rks(mol, grids, XC, dm)
    _, e_gpu32, _ = gpu_nr_rks(ctx32, ni, mol, grids, XC, dm)
    ctx_lift = build_context(mol, backend=backend, blksize=blksize, verify_ao=False,
                             dtype=F32, gemm_dtype=F64)
    _, e_gpulift, _ = gpu_nr_rks(ctx_lift, ni, mol, grids, XC, dm)

    # per-point breakdown of prod - ref
    d = e_pt['prod'] - e_pt['ref']
    r0 = rho['ref']
    lr = np.log10(np.maximum(r0[0], 1e-300))
    rho_edges = [-np.inf, -8, -6, -4, -2, 0, np.inf]
    rho_labels = ['<1e-8', '1e-8..1e-6', '1e-6..1e-4', '1e-4..1e-2', '1e-2..1', '>1']
    rho_key = np.digitize(lr, rho_edges[1:-1])
    ok = (r0[0] > 1e-10) & (r0[4] > 1e-14)
    z = np.full(npts, np.nan)
    z[ok] = ((r0[1:4, ok] ** 2).sum(axis=0) / (8.0 * r0[0, ok])) / r0[4, ok]
    z_labels = ['tiny rho/tau', 'z<0.5', '0.5..0.9', '0.9..0.99', '0.99..0.999', '>=0.999']
    z_key = np.zeros(npts, dtype=np.int64)
    z_key[ok] = 1 + np.digitize(z[ok], [0.5, 0.9, 0.99, 0.999])
    ia, dist = nearest_atom(grids.coords, mol.atom_coords())
    elem = np.array([mol.atom_pure_symbol(i) for i in range(mol.natm)])[ia]
    shell_edges = [0.3, 1.0, 2.0]
    shell_names = ['r<0.3', '0.3..1', '1..2', 'r>2']
    el_names = sorted(set(elem.tolist()))
    labels = ['%s %s' % (e, s) for e in el_names for s in shell_names]
    at_key = (np.searchsorted(el_names, elem) * len(shell_names)
              + np.digitize(dist, shell_edges))

    # pointwise relative errors of the prod density components
    big = r0[0] > 1e-6
    rel = {}
    for name, sl in (('rho', slice(0, 1)), ('grad', slice(1, 4)), ('tau', slice(4, 5))):
        num = np.abs(rho['prod'][sl, big] - r0[sl, big]).max(axis=0)
        den = np.abs(r0[sl, big]).max(axis=0) + 1e-300
        q = num / den
        rel[name] = {'median': float(np.median(q)), 'p99': float(np.quantile(q, 0.99)),
                     'max': float(q.max())}

    out = {
        'system': system, 'basis': 'def2-svp', 'xc': XC, 'backend': backend,
        'nao': int(mol.nao), 'natm': int(mol.natm), 'npts': int(npts),
        'grids_level': int(grids.level), 'dm_source': dm_source,
        'E_xc_ref': E['ref'],
        'dE_vs_ref': {k: E[k] - E['ref'] for k in E},
        'dnelec_vs_ref': {k: nelec[k] - nelec['ref'] for k in nelec},
        'checks': {'ref_minus_nr_rks': E['ref'] - e_cpu,
                   'prod_minus_gpu_nr_rks_f32': E['prod'] - e_gpu32,
                   'lift_gemm_minus_gpu_nr_rks_lift': E['lift_gemm'] - e_gpulift,
                   'nelec_nr_rks': n_cpu},
        'prod_pointwise': {'sum': math.fsum(d), 'abs_sum': math.fsum(np.abs(d)),
                           'rms_sum': math.sqrt(math.fsum(d * d)),
                           'by_log10_rho': binned(d, rho_key, rho_labels),
                           'by_z_tauW_over_tau': binned(d, z_key, z_labels),
                           'by_nearest_atom': binned(d, at_key, labels)},
        'prod_rel_err_rho_gt_1e-6': rel,
        'wall_s': {'density': t_density, 'total': time.perf_counter() - t0},
    }
    path = os.path.join(results_dir(), 'fp32_attribution_%s.json' % system)
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    return out, path


def report(out, path):
    print('== %s  nao=%d natm=%d npts=%d level=%d  E_xc(ref)=%.10f'
          % (out['system'], out['nao'], out['natm'], out['npts'],
             out['grids_level'], out['E_xc_ref']))
    print('   dm: %s' % out['dm_source'])
    for k, v in out['dE_vs_ref'].items():
        print('   %-10s dE=% .3e  dnelec=% .3e' % (k, v, out['dnelec_vs_ref'][k]))
    for k, v in out['checks'].items():
        print('   check %-34s % .3e' % (k, v))
    pp = out['prod_pointwise']
    print('   prod pointwise: sum=% .3e  sum|d|=%.3e  sqrt(sum d^2)=%.3e'
          % (pp['sum'], pp['abs_sum'], pp['rms_sum']))
    for group in ('by_log10_rho', 'by_z_tauW_over_tau', 'by_nearest_atom'):
        print('   %s:' % group)
        for lab, b in pp[group].items():
            if b['npts']:
                print('     %-14s sum=% .3e  sum|d|=%.3e  n=%d'
                      % (lab, b['sum'], b['abs_sum'], b['npts']))
    for k, v in out['prod_rel_err_rho_gt_1e-6'].items():
        print('   rel err %-5s median=%.2e p99=%.2e max=%.2e'
              % (k, v['median'], v['p99'], v['max']))
    print('   wall %.1fs (density %.1fs) -> %s'
          % (out['wall_s']['total'], out['wall_s']['density'], path))
    sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--systems', nargs='+', default=['water_dimer', 'water27', 'chain30'])
    ap.add_argument('--backend', default='cupy', choices=['cupy', 'numpy'])
    ap.add_argument('--blksize', type=int, default=8192)
    args = ap.parse_args(argv)
    for system in args.systems:
        report(*run(system, args.backend, args.blksize))


if __name__ == '__main__':
    main()
