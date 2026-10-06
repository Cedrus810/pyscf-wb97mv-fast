"""S5 r1: are the hi+lo remainders actually applied?  Pattern under test:
    S = fl32(sum x*c_hi) + fl32(sum x*c_lo)
S_hi already sits on the FP32 grid and |S_lo| < ulp(S_hi)/2 mostly, so the
sum rounds back to S_hi: the correction is dropped.  Compare the existing
code with its lo arrays zeroed, and with a per-product fma(x, c_hi, x*c_lo).
(1) AoEvaluator (cupy): pack lo arrays zeroed -> same X?
(2) dense int3c1e kernel: Mc_lo/MT_lo zeroed -> same A?
Grid jittered 1e-6 bohr (coherent rounding removed)."""
import numpy as np, importlib.util, os
def load(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), name + '.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
kba = load('k_bias_attribution')
from pyscf.sgx import sgx_jk
from pyscf_wb97mv_fast.gpu import backends, ao_eval
from pyscf_wb97mv_fast.gpu.shellpairs import build_shell_pairs, numpy_origin, int3c1e_reference
from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
cp = backends.cupy_module()
mol, sgx, dm = kba.setup(2)
w = np.asarray(sgx.grids.weights)
coords = np.asarray(sgx.grids.coords) + np.random.default_rng(7).normal(scale=1e-6, size=(len(w), 3))
origin = numpy_origin(mol.atom_coords().mean(axis=0)); rel = coords - origin
X64 = mol.eval_gto('GTOval_sph', coords)
nrm = (w[:, None] * X64 * X64).sum(axis=0)
def bias(Xv):
    b = (w[:, None] * X64 * (Xv - X64)).sum(axis=0) / nrm
    return 'b(O1s) % .2e  b(O2s) % .2e  b(O2p) % .2e  b(Od) % .2e  b(H1s) % .2e  |b|mean % .2e' % (
        b[0], b[1], b[3:6].mean(), b[9:14].mean(), b[14], np.abs(b).mean())
def X_with(zero):
    pack = ao_eval.ShellPack(mol, origin=origin)
    ev = ao_eval.AoEvaluator(mol, cp, pack=pack)
    for g in pack.groups:
        if 'exps' in zero: g.exps_lo_d = cp.zeros_like(g.exps_lo_d)
        if 'coeffs' in zero: g.coeffs_lo_d = cp.zeros_like(g.coeffs_lo_d)
    if 'T' in zero:
        orig = ao_eval._remainder32
        ao_eval._remainder32 = lambda a: np.zeros_like(np.asarray(a, dtype=np.float64))
    try:
        return cp.asnumpy(ev.eval(cp.asarray(rel), np.arange(mol.nbas), deriv=0)[0]).astype(np.float64)
    finally:
        if 'T' in zero: ao_eval._remainder32 = orig
print('(1) AoEvaluator, jittered grid: weighted per-AO relative bias')
for z in [(), ('exps',), ('coeffs',), ('T',), ('exps', 'coeffs', 'T')]:
    print('  lo zeroed: %-22s %s' % (','.join(z) or 'none (production)', bias(X_with(set(z)))))
print('  fl32(X64)                         ', bias(X64.astype(np.float32).astype(np.float64)))

print('(2) dense int3c1e kernel vs FP64 reference, 300 jittered grid points')
pairs = build_shell_pairs(mol)
sel = np.random.default_rng(3).choice(len(w), 300, replace=False)
for omega in (0.0, 0.3):
    ref = int3c1e_reference(pairs, coords[sel], omega=omega)
    big = np.abs(ref) > 1e-3 * np.abs(ref).max()
    for zero in (False, True):
        pd = pairs.to_device(cp, mem_budget=1 << 30, origin=origin)
        if zero:
            pd.sh_Mc_lo = cp.zeros_like(pd.sh_Mc_lo); pd.sh_MT_lo = cp.zeros_like(pd.sh_MT_lo)
        got = cp.asnumpy(int3c1e_fp32(cp, pd, cp.asarray(rel[sel]), omega=omega)).astype(np.float64)
        print('  omega=%.1f Mc/MT lo %-8s mean signed rel (big) % .3e' % (
            omega, 'zeroed' if zero else 'as is', np.mean((got[big] - ref[big]) / ref[big])))
