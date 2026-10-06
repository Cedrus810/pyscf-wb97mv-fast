"""S5 r1: term-by-term attribution of the GPU FP32 COSX K energy bias.

Water dimer, SGX level 2, init-guess dm (the test_sgx_k setup).  Reference:
the whole K pipeline in FP64 NumPy (int3c1e_reference, eval_gto, FP64
contractions; matches CPU get_k_only to ~1e-14).  Then

  stage A (pipeline):  swap ONE ingredient for its GPU FP32 version
                       (A from the dense FP32 kernel, X from AoEvaluator,
                       fl32(proj dm)) and report dE = 1/4 tr(D dK);
  stage B (integral):  FP64 reference with ONE kernel constant rounded to
                       FP32 the way the kernel does it (sqrt(theta),
                       -2 p theta, prefactor, E, T, Boys FP32, coordinates
                       in the FP32 origin frame, Mc/MT without hi+lo).

The FP64 reference is a Python loop over pairs with einsum (single
threaded), so the grid is split into chunks and farmed out to a spawn
process pool (OMP_NUM_THREADS=1 per worker); each chunk returns its K
contribution and the parent sums them in chunk order (deterministic).  The
Boys-FP32 variant needs CUDA in the workers and uses a small separate pool.

Usage:  python -B benchmarks/sgx/k_bias_attribution.py [--omega 0.3 0.0]
        [--nproc 72] [--gpu-nproc 4]
"""
import argparse
import copy
import multiprocessing as mp
import os
import time

import numpy as np

f32 = lambda a: np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)

_W = {}          # worker state


def _winit(data):
    _W.update(data)


def _variant_pairs(spec):
    from pyscf_wb97mv_fast.gpu import shellpairs  # noqa: F401
    key = spec.get('pairs')
    cache = _W.setdefault('_pairs_cache', {})
    if key not in cache:
        base = _W['pairs']
        q = copy.copy(base)
        if key == 'prefac':
            q.prim_prefac = f32(base.prim_prefac)
        elif key == 'E':
            q.prim_E = f32(base.prim_E)
        elif key == 'McMT':
            q.sh_Mc = f32(base.sh_Mc)
            q.sh_MT = f32(base.sh_MT)
        elif key == 'frame':
            q.prim_P = base.prim_P - _W['origin']
        cache[key] = q
    return cache[key]


def _wtask(args):
    spec, i0, i1 = args
    from pyscf_wb97mv_fast.gpu import shellpairs
    pairs = _variant_pairs(spec)
    if spec.get('pairs') == 'frame':
        coords = f32(_W['coords'][i0:i1] - _W['origin'])
    else:
        coords = _W['coords'][i0:i1]
    orig_fill, orig_boys = shellpairs._fill_R, shellpairs._boys_ref

    def fill(T, pth, sth, d):
        if spec.get('round_T'):
            T = f32(T)
        if spec.get('round_pth'):
            pth = float(f32(pth))
        if spec.get('round_sth'):
            sth = float(f32(sth))
        return orig_fill(T, pth, sth, d)

    boys = spec.get('boys')
    if boys == 'f32T':
        shellpairs._boys_ref = lambda n, T: orig_boys(n, f32(T))
    elif boys == 'gpu':
        from pyscf_wb97mv_fast.gpu import backends
        from pyscf_wb97mv_fast.gpu.boys import boys_eval
        cp = backends.cupy_module()
        memo = {}

        def gpu_boys(n, T):
            if memo.get('T') is not T:
                memo['T'] = T
                memo['F'] = cp.asnumpy(boys_eval(
                    cp, cp.asarray(np.asarray(T, np.float32)), 4)).astype(np.float64)
            return memo['F'][n]
        shellpairs._boys_ref = gpu_boys
    shellpairs._fill_R = fill
    try:
        A = shellpairs.int3c1e_reference(pairs, coords, omega=_W['omega'])
    finally:
        shellpairs._fill_R, shellpairs._boys_ref = orig_fill, orig_boys
    X = _W['X32'] if spec.get('X') == '32' else _W['X64']
    pdm = _W['pdm32'] if spec.get('pdm') == '32' else _W['pdm64']
    Xc = X[i0:i1]
    G = _W['w'][i0:i1, None] * np.einsum('gmn,gm->gn', A, Xc @ pdm)
    return Xc.T @ G


def setup(level=2):
    from pyscf import dft
    from pyscf_wb97mv_fast.core.testsystems import build_mol
    mol = build_mol('water_dimer', 'def2-svp')
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    sgx = mf.with_df
    sgx.grids_level_i = sgx.grids_level_f = level
    sgx.build(level=level)
    dm = mf.get_init_guess()
    dm = (dm + dm.T) / 2
    return mol, sgx, dm


def run(omega, cp, level, nproc, gpu_nproc, chunk):
    from pyscf.sgx import sgx_jk
    from pyscf_wb97mv_fast.gpu.shellpairs import build_shell_pairs, numpy_origin
    from pyscf_wb97mv_fast.gpu.ao_eval import AoEvaluator, ShellPack
    from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu

    t0 = time.perf_counter()
    mol, sgx, dm = setup(level)
    with mol.with_range_coulomb(omega):
        k_cpu = sgx_jk.get_k_only(sgx, dm, hermi=1)
        k_gpu = get_k_only_gpu(GpuKBuilder(mol, cp, tile_tol=0.0), sgx, dm, hermi=1)
    proj = sgx._pjs_data._overlap_correction_matrix
    sym = sgx._pjs_data.sym_ovlp
    coords = np.asarray(sgx.grids.coords, dtype=np.float64)
    w = np.asarray(sgx.grids.weights, dtype=np.float64)
    origin = numpy_origin(mol.atom_coords().mean(axis=0))
    pairs = build_shell_pairs(mol)
    X64 = mol.eval_gto('GTOval_sph', coords)
    ao = AoEvaluator(mol, cp, pack=ShellPack(mol, origin=origin))
    X32 = cp.asnumpy(ao.eval(cp.asarray(coords - origin), np.arange(mol.nbas),
                             deriv=0)[0]).astype(np.float64)
    pdm64 = proj @ dm
    ng = len(w)
    bounds = [(i0, min(i0 + chunk, ng)) for i0 in range(0, ng, chunk)]

    def post(K):
        if sym:
            K = proj.T @ K
        return (K + K.T) * 0.5
    e = lambda K: 0.25 * np.einsum('ij,ji->', dm, K)
    print('omega=%.2f  ng=%d  nao=%d  npairs=%d  nprims=%d  E_K(cpu)=%.10f  chunks=%d'
          % (omega, ng, mol.nao, pairs.npairs, pairs.nprims, e(k_cpu), len(bounds)), flush=True)

    # A from the dense FP32 GPU kernel, everything else FP64 (parent, GPU)
    pairs_dev = pairs.to_device(cp, mem_budget=1 << 30, origin=origin)
    F64 = X64 @ pdm64
    K = np.zeros((mol.nao, mol.nao))
    for i0, i1 in bounds:
        A = cp.asnumpy(int3c1e_fp32(cp, pairs_dev, cp.asarray(coords[i0:i1] - origin),
                                    omega=omega)).astype(np.float64)
        K += X64[i0:i1].T @ (w[i0:i1, None] * np.einsum('gmn,gm->gn', A, F64[i0:i1]))
    k_Agpu = post(K)

    pairs_ship = copy.copy(pairs)
    pairs_ship.mol = None
    data = dict(pairs=pairs_ship, coords=coords, w=w, origin=origin, omega=omega,
                X64=X64, X32=X32, pdm64=pdm64, pdm32=f32(pdm64))
    ctx = mp.get_context('spawn')
    variants = [('ref', {})]
    variants += [('A: X from AoEvaluator', {'X': '32'}),
                 ('A: fl32(proj dm)', {'pdm': '32'})]
    if omega > 0:
        variants.append(('B: fl32(sqrt theta)', {'round_sth': True}))
    variants += [('B: fl32(-2 p theta)', {'round_pth': True}),
                 ('B: fl32(T)', {'round_T': True}),
                 ('B: exact Boys at fl32(T)', {'boys': 'f32T'}),
                 ('B: fl32(prefactor K 2pi/p)', {'pairs': 'prefac'}),
                 ('B: fl32(E Hermite)', {'pairs': 'E'}),
                 ('B: fl32(Mc/MT) plain [now hi+lo]', {'pairs': 'McMT'}),
                 ('B: fl32(coords, origin frame)', {'pairs': 'frame'})]
    Ks = {}
    with ctx.Pool(nproc, initializer=_winit, initargs=(data,)) as pool:
        for name, spec in variants:
            parts = pool.map(_wtask, [(spec, i0, i1) for i0, i1 in bounds], chunksize=1)
            Ks[name] = post(sum(parts))
            print('  [%s  %.0fs]' % (name, time.perf_counter() - t0), flush=True)
    with ctx.Pool(gpu_nproc, initializer=_winit, initargs=(data,)) as pool:
        parts = pool.map(_wtask, [({'boys': 'gpu'}, i0, i1) for i0, i1 in bounds], chunksize=1)
        Ks['B: GPU Boys at fl32(T)'] = post(sum(parts))
        print('  [GPU Boys  %.0fs]' % (time.perf_counter() - t0), flush=True)

    k_ref = Ks.pop('ref')
    k_bT = Ks.pop('B: exact Boys at fl32(T)')
    k_bG = Ks.pop('B: GPU Boys at fl32(T)')
    rows = [('CPU get_k_only', k_cpu, k_ref), ('GPU fused (total)', k_gpu, k_ref),
            ('A: integrals FP32 kernel', k_Agpu, k_ref)]
    rows += [(n, Kv, k_ref) for n, Kv in Ks.items()]
    rows += [('B: Boys FP32 kernel (vs exact, fl32 T)', k_bG, k_bT)]
    print('  %-40s %12s %10s' % ('variant', 'dE vs FP64', 'max|dK|'))
    for name, Kv, Kr in rows:
        print('  %-40s %12.3e %10.2e' % (name, e(Kv) - e(Kr), np.abs(Kv - Kr).max()))
    print('  [done %.0fs]' % (time.perf_counter() - t0), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--omega', type=float, nargs='+', default=[0.3, 0.0])
    ap.add_argument('--level', type=int, default=2)
    ap.add_argument('--nproc', type=int, default=72)
    ap.add_argument('--gpu-nproc', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=128)
    a = ap.parse_args()
    from pyscf_wb97mv_fast.gpu import backends
    cp = backends.cupy_module()
    # spawned workers read this at their numpy import; the parent's BLAS is
    # already initialized and keeps its threads
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    for om in a.omega:
        run(om, cp, a.level, a.nproc, a.gpu_nproc, a.chunk)
