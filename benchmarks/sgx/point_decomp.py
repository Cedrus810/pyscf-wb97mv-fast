"""S5 r1: per-grid-point split of the integral-kernel energy error,
e_g = 1/4 w_g F_g^T (A32_g - A64_g) F_g  (F = X proj dm, X FP64; sum_g e_g
is the 'A: integrals FP32 kernel' row of k_bias_attribution), binned by the
distance to the nearest nucleus.  A32 = current dense FP32 GPU kernel, A64
= int3c1e_reference (spawn pool, single-threaded workers).  Also the same
split for the FP64 reference at FP32-rounded coordinates (origin frame)."""
import argparse, os, time, multiprocessing as mp
import numpy as np
_W = {}
def _init(d): _W.update(d)
def _task(args):
    i0, i1, frame = args
    from pyscf_wb97mv_fast.gpu.shellpairs import int3c1e_reference
    pairs, c = _W['pairs'], _W['coords'][i0:i1]
    if frame:
        import copy
        pairs = copy.copy(pairs); pairs.prim_P = _W['pairs'].prim_P - _W['origin']
        c = (c - _W['origin']).astype(np.float32).astype(np.float64)
    A = int3c1e_reference(pairs, c, omega=_W['omega'])
    F = _W['F'][i0:i1]
    return 0.25 * _W['w'][i0:i1] * np.einsum('gm,gmn,gn->g', F, A, F)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--nproc', type=int, default=72)
    ap.add_argument('--jitter', type=float, default=0.0, help='grid jitter (bohr): removes coherent rounding')
    a = ap.parse_args()
    import importlib.util, copy
    s = importlib.util.spec_from_file_location('kba', os.path.join(os.path.dirname(__file__), 'k_bias_attribution.py'))
    kba = importlib.util.module_from_spec(s); s.loader.exec_module(kba)
    from pyscf.sgx import sgx_jk
    from pyscf_wb97mv_fast.gpu import backends
    from pyscf_wb97mv_fast.gpu.shellpairs import build_shell_pairs, numpy_origin
    from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
    cp = backends.cupy_module()
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    for omega in (0.0, 0.3):
        t0 = time.perf_counter()
        mol, sgx, dm = kba.setup(2)
        with mol.with_range_coulomb(omega):
            sgx_jk.get_k_only(sgx, dm, hermi=1)
        proj = sgx._pjs_data._overlap_correction_matrix
        coords = np.asarray(sgx.grids.coords); w = np.asarray(sgx.grids.weights)
        if a.jitter:
            coords = coords + np.random.default_rng(7).normal(scale=a.jitter, size=coords.shape)
        origin = numpy_origin(mol.atom_coords().mean(axis=0))
        pairs = build_shell_pairs(mol); pd = pairs.to_device(cp, mem_budget=1 << 30, origin=origin)
        F = mol.eval_gto('GTOval_sph', coords) @ (proj @ dm)
        e32 = np.empty(len(w))
        for i0 in range(0, len(w), 1024):
            i1 = min(i0 + 1024, len(w))
            A = cp.asnumpy(int3c1e_fp32(cp, pd, cp.asarray(coords[i0:i1] - origin), omega=omega)).astype(np.float64)
            e32[i0:i1] = 0.25 * w[i0:i1] * np.einsum('gm,gmn,gn->g', F[i0:i1], A, F[i0:i1])
        ps = copy.copy(pairs); ps.mol = None
        bounds = [(i, min(i + 128, len(w))) for i in range(0, len(w), 128)]
        with mp.get_context('spawn').Pool(a.nproc, initializer=_init,
                                          initargs=(dict(pairs=ps, coords=coords, w=w, F=F, omega=omega, origin=origin),)) as pool:
            e64 = np.concatenate(pool.map(_task, [(i0, i1, False) for i0, i1 in bounds], chunksize=1))
            efr = np.concatenate(pool.map(_task, [(i0, i1, True) for i0, i1 in bounds], chunksize=1))
        d = np.linalg.norm(coords[:, None, :] - mol.atom_coords()[None], axis=2)
        dn = d.min(axis=1); near = d.argmin(axis=1)
        isO = np.array([mol.atom_symbol(k) == 'O' for k in range(mol.natm)])[near]
        de, dfr = e32 - e64, efr - e64
        print('jitter=%.0e  omega=%.1f  sum dE: GPU kernel % .3e   fl32 coords (FP64 ref) % .3e   [%.0fs]'
              % (a.jitter, omega, de.sum(), dfr.sum(), time.perf_counter() - t0))
        edges = [0, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 1e9]
        print('   r_nearest bin         npts   GPU-kernel dE  (O/H atoms)          fl32-coords dE')
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (dn >= lo) & (dn < hi)
            print('   [%7.0e,%7.0e) %6d   % .3e  (% .2e / % .2e)   % .3e' % (
                lo, hi, m.sum(), de[m].sum(), de[m & isO].sum(), de[m & ~isO].sum(), dfr[m].sum()))
        np.save('benchmarks/results/s5r1_point_decomp_omega%.1f_jit%.0e.npy' % (omega, a.jitter), np.stack([dn, de, dfr, w]))
