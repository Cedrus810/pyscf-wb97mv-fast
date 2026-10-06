"""S5 r1: per-stage attribution of the genuine (non-coherent) FP32 X bias:
the x_bias_attribution emulation variants on a grid jittered by 1e-6 bohr
(coherent rounding removed, see x_coherence_check)."""
import numpy as np, importlib.util, os
def load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), name + '.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
kba = load('k_bias_attribution'); xba = load('x_bias_attribution')
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.gpu import backends, ao_eval
from pyscf_wb97mv_fast.gpu.shellpairs import build_shell_pairs, numpy_origin
from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
cp = backends.cupy_module()
for omega in (0.0, 0.3):
    mol, sgx, dm = kba.setup(2)
    with mol.with_range_coulomb(omega):
        sgx_jk.get_k_only(sgx, dm, hermi=1)
    proj = sgx._pjs_data._overlap_correction_matrix; sym = sgx._pjs_data.sym_ovlp
    w = np.asarray(sgx.grids.weights)
    coords = np.asarray(sgx.grids.coords) + np.random.default_rng(7).normal(scale=1e-6, size=(len(w), 3))
    origin = numpy_origin(mol.atom_coords().mean(axis=0)); rel = coords - origin
    pd = build_shell_pairs(mol).to_device(cp, mem_budget=1 << 30, origin=origin)
    A = [cp.asnumpy(int3c1e_fp32(cp, pd, cp.asarray(rel[i:i+1024]), omega=omega)).astype(np.float64)
         for i in range(0, len(w), 1024)]
    pdm = proj @ dm
    def eK(X):
        F = X @ pdm; K = np.zeros((mol.nao, mol.nao))
        for k, i0 in enumerate(range(0, len(w), 1024)):
            i1 = min(i0 + 1024, len(w))
            K += X[i0:i1].T @ (w[i0:i1, None] * np.einsum('gmn,gm->gn', A[k], F[i0:i1]))
        if sym: K = proj.T @ K
        return 0.25 * np.einsum('ij,ji->', dm, (K + K.T) * 0.5)
    X64 = mol.eval_gto('GTOval_sph', coords); e64 = eK(X64)
    nrm = (w[:, None] * X64 * X64).sum(axis=0)
    print('omega=%.2f (grid jittered 1e-6 bohr)' % omega)
    ev = ao_eval.AoEvaluator(mol, cp, pack=ao_eval.ShellPack(mol, origin=origin))
    rows = [('cupy FP32 (production)', cp.asnumpy(ev.eval(cp.asarray(rel), np.arange(mol.nbas), deriv=0)[0]).astype(np.float64))]
    for st in [(), ('d',), ('arg',), ('gv',), ('ctr',), ('c2s',), ('d', 'arg'), ('ctr', 'c2s'),
               ('d', 'arg', 'gv'), ('d', 'arg', 'gv', 'ctr'), ('d', 'arg', 'gv', 'ctr', 'c2s')]:
        rows.append(('emu, FP64 stages %s' % (','.join(st) or '-'), xba.numpy_X(mol, origin, rel, xba.make_chunk_eval(set(st)))))
    for name, Xv in rows:
        b = (w[:, None] * X64 * (Xv - X64)).sum(axis=0) / nrm
        print('  %-40s dE_K % .3e   b(O1s) % .2e  b(O2s) % .2e  b(O2p) % .2e  b(H1s) % .2e'
              % (name, eK(Xv) - e64, b[0], b[1], b[3:6].mean(), b[14]), flush=True)
