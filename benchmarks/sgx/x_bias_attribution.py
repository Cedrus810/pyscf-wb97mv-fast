"""S5 r1: where does the systematic bias of the FP32 AO values X come from?

k_bias_attribution.py showed X from AoEvaluator carries 1.54e-7 of the
1.64e-7 long-range K energy bias on the water dimer.  Here A is fixed (the
dense FP32 GPU integrals; the A error cancels in the difference) and only X
changes:  dE(Xv) = e(K(Xv, A)) - e(K(X64, A)), e = 1/4 tr(D K), with X used
both in F = X proj dm and in K = X^T G.

Variants of X: the production cupy FP32 path, the same AoEvaluator code run
with numpy at FP32, fl32(X64) (output rounding alone), and the numpy FP32
path with single stages done in FP64 instead (see STAGES).  Also printed:
the weighted per-AO relative bias b_mu = sum_g w X64 dX / sum_g w X64^2.

Usage:  python -B benchmarks/sgx/x_bias_attribution.py [--omega 0.3 0.0]
"""
import argparse
import time

import numpy as np

from pyscf_wb97mv_fast.gpu import ao_eval
from pyscf_wb97mv_fast.gpu.precision import FP32, FP64

f32 = lambda a: np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)


def make_chunk_eval(stage64):
    """A copy of AoEvaluator._eval_shell_chunk (deriv=0, numpy) where the
    stages named in stage64 run in FP64 and are rounded to FP32 afterwards:
      'd'     coordinate differences and r2
      'arg'   the exponent argument -a r2
      'gv'    E * monomials (Cartesian primitive values)
      'ctr'   the contraction einsum
      'c2s'   the cart->sph transform
    """
    def chunk(self, coords, l, nctr, centers, exps, coeffs, T, deriv,
              exps_lo=None, coeffs_lo=None, T_lo=None):
        assert deriv == 0
        npts = coords.shape[0]
        Sc = centers.shape[0]
        comps = ao_eval.cart_components(l)
        ncart = len(comps)
        a64 = exps.astype(FP64) + (0 if exps_lo is None else exps_lo.astype(FP64))
        c64 = coeffs.astype(FP64) + (0 if coeffs_lo is None else coeffs_lo.astype(FP64))
        T64 = T.astype(FP64) + (0 if T_lo is None else T_lo.astype(FP64))
        d64 = coords[:, None, :] - centers[None, :, :]
        if 'd' in stage64:
            d = d64
            r2 = (d64 * d64).sum(axis=2)
        else:
            d = d64.astype(FP32)
            r2 = (d * d).sum(axis=2)
        if 'arg' in stage64:
            E = np.exp(-a64[None] * r2.astype(FP64)[:, :, None])
        else:
            arg = -exps[None] * r2.astype(FP32)[:, :, None]
            E = np.exp(arg.astype(FP64)).astype(FP32)
            E = E - (E * exps_lo[None]) * r2.astype(FP32)[:, :, None]
        hi = 'gv' in stage64
        if not hi:
            E = E.astype(FP32)
            d = d.astype(FP32)
        val = np.empty((npts, Sc, nctr, ncart), dtype=FP64 if hi else FP32)
        for c, (lx, ly, lz) in enumerate(comps):
            gv = E * (d[:, :, 0] ** lx)[:, :, None] * (d[:, :, 1] ** ly)[:, :, None] \
                * (d[:, :, 2] ** lz)[:, :, None]
            if 'ctr' in stage64:
                cv = np.einsum('psk,sck->psc', gv.astype(FP64), c64)
            else:
                gv = gv.astype(FP32)
                cv = np.einsum('psk,sck->psc', gv, coeffs) \
                    + np.einsum('psk,sck->psc', gv, coeffs_lo)
            val[:, :, :, c] = cv.reshape(npts, Sc, nctr)
        lead = val.reshape(-1, ncart)
        if 'c2s' in stage64:
            out = lead.astype(FP64) @ T64
        else:
            lead = lead.astype(FP32)
            out = lead @ T + lead @ T_lo
        out = out.astype(FP32)
        return out.reshape(npts, -1)[None]
    return chunk


def numpy_X(mol, origin, coords_rel, chunk_fn=None):
    pack = ao_eval.ShellPack(mol, origin=origin)
    ev = ao_eval.AoEvaluator(mol, np, pack=pack, dtype=FP32, verify=False)
    if chunk_fn is not None:
        ev._eval_shell_chunk = chunk_fn.__get__(ev)
    out = np.empty((coords_rel.shape[0], mol.nao))
    for i0 in range(0, coords_rel.shape[0], 2048):
        i1 = min(i0 + 2048, coords_rel.shape[0])
        out[i0:i1] = ev.eval(coords_rel[i0:i1], np.arange(mol.nbas), deriv=0)[0]
    return out


def run(omega, cp, level):
    from pyscf.sgx import sgx_jk
    from pyscf_wb97mv_fast.gpu.shellpairs import build_shell_pairs, numpy_origin
    from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        'kba', os.path.join(os.path.dirname(__file__), 'k_bias_attribution.py'))
    kba = importlib.util.module_from_spec(spec); spec.loader.exec_module(kba)

    t0 = time.perf_counter()
    mol, sgx, dm = kba.setup(level)
    with mol.with_range_coulomb(omega):
        sgx_jk.get_k_only(sgx, dm, hermi=1)
    proj = sgx._pjs_data._overlap_correction_matrix
    sym = sgx._pjs_data.sym_ovlp
    coords = np.asarray(sgx.grids.coords, dtype=np.float64)
    w = np.asarray(sgx.grids.weights, dtype=np.float64)
    origin = numpy_origin(mol.atom_coords().mean(axis=0))
    rel = coords - origin
    pairs = build_shell_pairs(mol)
    pairs_dev = pairs.to_device(cp, mem_budget=1 << 30, origin=origin)
    pdm = proj @ dm
    A = []
    for i0 in range(0, len(w), 1024):
        A.append(cp.asnumpy(int3c1e_fp32(cp, pairs_dev, cp.asarray(rel[i0:i0 + 1024]),
                                         omega=omega)).astype(np.float64))

    def eK(X):
        F = X @ pdm
        K = np.zeros((mol.nao, mol.nao))
        for k, i0 in enumerate(range(0, len(w), 1024)):
            i1 = min(i0 + 1024, len(w))
            G = w[i0:i1, None] * np.einsum('gmn,gm->gn', A[k], F[i0:i1])
            K += X[i0:i1].T @ G
        if sym:
            K = proj.T @ K
        K = (K + K.T) * 0.5
        return 0.25 * np.einsum('ij,ji->', dm, K)

    X64 = mol.eval_gto('GTOval_sph', coords)
    e64 = eK(X64)
    ev = ao_eval.AoEvaluator(mol, cp, pack=ao_eval.ShellPack(mol, origin=origin))
    Xgpu = cp.asnumpy(ev.eval(cp.asarray(rel), np.arange(mol.nbas), deriv=0)[0]).astype(np.float64)
    print('omega=%.2f  ng=%d  E_K(X64, A32)=%.10f' % (omega, len(w), e64), flush=True)
    variants = [('cupy FP32 (production)', Xgpu),
                ('fl32(X64) output rounding only', f32(X64)),
                ('numpy FP32, AoEvaluator code', numpy_X(mol, origin, rel))]
    for st in [(), ('d',), ('arg',), ('gv',), ('ctr',), ('c2s',),
               ('d', 'arg'), ('d', 'arg', 'gv', 'ctr', 'c2s')]:
        variants.append(('numpy FP32 emu, FP64 stages %s' % (','.join(st) or '-'),
                         numpy_X(mol, origin, rel, make_chunk_eval(set(st)))))
    nrm = (w[:, None] * X64 * X64).sum(axis=0)
    print('  %-44s %12s %10s %12s' % ('X variant', 'dE_K', 'max|dX|', 'mean b_mu'))
    for name, Xv in variants:
        dX = Xv - X64
        b = (w[:, None] * X64 * dX).sum(axis=0) / nrm
        print('  %-44s %12.3e %10.2e %12.3e' % (name, eK(Xv) - e64, np.abs(dX).max(), b.mean()),
              flush=True)
    # per-AO bias of the production path, by shell
    b = (w[:, None] * X64 * (Xgpu - X64)).sum(axis=0) / nrm
    labels = mol.ao_labels()
    print('  per-AO weighted relative bias b_mu of the cupy path:')
    for mu in range(mol.nao):
        print('    %3d %-18s % .3e' % (mu, labels[mu], b[mu]))
    print('  [done %.0fs]' % (time.perf_counter() - t0), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--omega', type=float, nargs='+', default=[0.3, 0.0])
    ap.add_argument('--level', type=int, default=2)
    a = ap.parse_args()
    from pyscf_wb97mv_fast.gpu import backends
    cp = backends.cupy_module()
    for om in a.omega:
        run(om, cp, a.level)
