"""S5 r1: is the FP32 X bias coherent rounding on the atom-centred grid?

(1) jitter every grid point by ~1e-6 bohr (exact X recomputed there): if the
    bias is coherent rounding of one-centre values along each radial shell
    (identical for every same-element atom), decorrelating the points must
    collapse it to the random level;
(2) per-radial-shell spread of the O 1s relative error (constant within a
    shell = coherent);
(3) with exact X: fl32(F) alone, fl32(G) alone, fl32(proj dm) hi+lo.
A fixed = dense FP32 GPU integrals (cancels in the differences)."""
import numpy as np, importlib.util, os, sys
sys.path.insert(0, os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location('kba', os.path.join(os.path.dirname(__file__), 'k_bias_attribution.py'))
kba = importlib.util.module_from_spec(spec); spec.loader.exec_module(kba)
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.gpu import backends, ao_eval
from pyscf_wb97mv_fast.gpu.shellpairs import build_shell_pairs, numpy_origin
from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
f32 = kba.f32
cp = backends.cupy_module()
for omega in (0.3, 0.0):
    mol, sgx, dm = kba.setup(2)
    with mol.with_range_coulomb(omega):
        sgx_jk.get_k_only(sgx, dm, hermi=1)
    proj = sgx._pjs_data._overlap_correction_matrix; sym = sgx._pjs_data.sym_ovlp
    coords = np.asarray(sgx.grids.coords, dtype=np.float64); w = np.asarray(sgx.grids.weights)
    origin = numpy_origin(mol.atom_coords().mean(axis=0))
    pd = build_shell_pairs(mol).to_device(cp, mem_budget=1 << 30, origin=origin)
    A = [cp.asnumpy(int3c1e_fp32(cp, pd, cp.asarray(coords[i:i+1024] - origin), omega=omega)).astype(np.float64)
         for i in range(0, len(w), 1024)]
    pdm = proj @ dm
    def eK(X, pdm=pdm, rF=False, rG=False):
        F = X @ pdm
        if rF: F = f32(F)
        K = np.zeros((mol.nao, mol.nao))
        for k, i0 in enumerate(range(0, len(w), 1024)):
            i1 = min(i0 + 1024, len(w))
            G = w[i0:i1, None] * np.einsum('gmn,gm->gn', A[k], F[i0:i1])
            if rG: G = f32(G)
            K += X[i0:i1].T @ G
        if sym: K = proj.T @ K
        K = (K + K.T) * 0.5
        return 0.25 * np.einsum('ij,ji->', dm, K)
    ev = ao_eval.AoEvaluator(mol, cp, pack=ao_eval.ShellPack(mol, origin=origin))
    gpuX = lambda c: cp.asnumpy(ev.eval(cp.asarray(c - origin), np.arange(mol.nbas), deriv=0)[0]).astype(np.float64)
    X64 = mol.eval_gto('GTOval_sph', coords); e64 = eK(X64)
    print('omega=%.2f' % omega)
    print('  %-46s % .3e' % ('Xgpu (grid as is)', eK(gpuX(coords)) - e64))
    print('  %-46s % .3e' % ('fl32(X64) (grid as is)', eK(f32(X64)) - e64))
    rng = np.random.default_rng(7)
    for s in (1e-6, 1e-4):
        cj = coords + rng.normal(scale=s, size=coords.shape)
        Xj64 = mol.eval_gto('GTOval_sph', cj); ej = eK(Xj64)
        print('  %-46s % .3e' % ('Xgpu, grid jittered %.0e bohr' % s, eK(gpuX(cj)) - ej))
        print('  %-46s % .3e' % ('fl32(X64), grid jittered %.0e bohr' % s, eK(f32(Xj64)) - ej))
    print('  %-46s % .3e' % ('X64, fl32(F) only', eK(X64, rF=True) - e64))
    print('  %-46s % .3e' % ('X64, fl32(G) only', eK(X64, rG=True) - e64))
    print('  %-46s % .3e' % ('X64, fl32(proj dm)', eK(X64, pdm=f32(pdm)) - e64))
    print('  %-46s % .3e' % ('X64, proj dm hi+lo', eK(X64, pdm=f32(pdm) + f32(pdm - f32(pdm))) - e64))
    if omega == 0.3:
        # (2) O 1s (AO 0) relative error on atom 0's own grid points, by radius
        r = np.linalg.norm(coords - mol.atom_coord(0), axis=1)
        own = np.flatnonzero(r < 0.6)
        rel = (gpuX(coords)[own, 0] - X64[own, 0]) / X64[own, 0]
        key = np.round(r[own], 9)
        uniq = np.unique(key)
        within = np.mean([rel[key == u].std() for u in uniq if (key == u).sum() > 10])
        means = np.array([rel[key == u].mean() for u in uniq if (key == u).sum() > 10])
        print('  O1 1s on atom-0 shells r<0.6: %d shells, mean within-shell std %.2e, across-shell std of means %.2e, mean %.2e'
              % (len(means), within, means.std(), means.mean()))
