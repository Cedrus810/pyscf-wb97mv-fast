"""S5 r1: energy effect of fl32(w) (SGX weights) on K, A = dense FP32 GPU
integrals, X FP64; plus the grid-jitter control."""
import numpy as np, importlib.util, os
s = importlib.util.spec_from_file_location('kba', os.path.join(os.path.dirname(__file__), 'k_bias_attribution.py'))
kba = importlib.util.module_from_spec(s); s.loader.exec_module(kba)
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.gpu import backends
from pyscf_wb97mv_fast.gpu.shellpairs import build_shell_pairs, numpy_origin
from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
cp = backends.cupy_module(); f32 = kba.f32
for omega in (0.0, 0.3):
    mol, sgx, dm = kba.setup(2)
    with mol.with_range_coulomb(omega):
        sgx_jk.get_k_only(sgx, dm, hermi=1)
    proj = sgx._pjs_data._overlap_correction_matrix; sym = sgx._pjs_data.sym_ovlp
    coords = np.asarray(sgx.grids.coords); w = np.asarray(sgx.grids.weights)
    origin = numpy_origin(mol.atom_coords().mean(axis=0))
    pd = build_shell_pairs(mol).to_device(cp, mem_budget=1 << 30, origin=origin)
    A = [cp.asnumpy(int3c1e_fp32(cp, pd, cp.asarray(coords[i:i+1024] - origin), omega=omega)).astype(np.float64)
         for i in range(0, len(w), 1024)]
    X = mol.eval_gto('GTOval_sph', coords); F = X @ (proj @ dm)
    def eK(wv):
        K = np.zeros((mol.nao, mol.nao))
        for k, i0 in enumerate(range(0, len(w), 1024)):
            i1 = min(i0 + 1024, len(w))
            K += X[i0:i1].T @ (wv[i0:i1, None] * np.einsum('gmn,gm->gn', A[k], F[i0:i1]))
        if sym: K = proj.T @ K
        return 0.25 * np.einsum('ij,ji->', dm, (K + K.T) * 0.5)
    rel = (f32(w) - w) / w
    print('omega=%.1f  dE fl32(w) % .3e   sum w*relerr/sum w % .2e   distinct w values %d of %d'
          % (omega, eK(f32(w)) - eK(w), (w * rel).sum() / w.sum(), len(np.unique(w)), len(w)))
