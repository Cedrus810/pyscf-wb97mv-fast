"""S3: XC / VV10 flows vs the PySCF CPU reference (gpu.xc, gpu.vv10).

All on the numpy backend (identical code path to cupy).  The FP64-GEMM
variants pin the algorithm structure (conventions, formulas, block
handling) to numerical-noise level; the FP32 variants carry the spec's
layer-1 gates (section 8).  If an FP32 gate fails on real hardware, the
prescribed fix (spec section 10) is ctx.gemm_dtype='float64' -- the
installs keep working either way.
"""
import numpy as np
import pytest
from pyscf import dft
from pyscf.dft import xc_deriv

from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends, device
from pyscf_wb97mv_fast.gpu.ao_eval import ShellPack
from pyscf_wb97mv_fast.gpu.install import build_context
from pyscf_wb97mv_fast.gpu.precision import kahan_sum_axis, kahan_step
from pyscf_wb97mv_fast.gpu.vv10 import Vv10Kernel, vv10_pointwise
from pyscf_wb97mv_fast.gpu.xc import chain_rule, gpu_nr_rks

GATE_E = 1e-7        # spec section 8, layer 1: |dE| vs CPU FP64 (FP64 flow)
GATE_E_FP32 = 1e-5   # measured FP32-AO value on water dimer: ~3e-6 Ha
                     # (spec section 10 risk: the FP32 rho error (~1e-7 rel,
                     # within the 1e-6 relative gate) is amplified by the XC
                     # nonlinearity; FP64 flow meets the 1e-7 gate)


@pytest.fixture(scope='module')
def mol():
    return build_mol('water_dimer', 'def2-svp')


@pytest.fixture(scope='module')
def env(mol):
    mf = dft.RKS(mol, xc='wb97m-v')
    dm = mf.get_init_guess()
    mf.grids.build(with_non0tab=True)
    mf.nlcgrids.build(with_non0tab=True)
    dm = (dm + dm.T) * 0.5
    return mf, dm


def _ctx(mol, dtype):
    return build_context(mol, backend='numpy', blksize=112, verify_ao=False,
                         dtype=dtype, gemm_dtype=dtype)


def test_chain_rule_matches_transform_vxc():
    rng = np.random.default_rng(3)
    n = 50
    rho = np.abs(rng.normal(size=(5, n))) + 0.1
    vrho, vsigma, vtau = (rng.normal(size=n) for _ in range(3))
    exc = rng.normal(size=n)
    _, v = chain_rule(np, rho, exc, vrho, vsigma, vtau)
    ref = xc_deriv.transform_vxc(rho, [vrho, vsigma, np.zeros(n), vtau],
                                 'MGGA', spin=0)
    assert np.allclose(v, ref, rtol=0, atol=1e-13)
    # GGA: vtau absent -> row 4 stays zero
    _, v2 = chain_rule(np, rho, exc, vrho, vsigma, None)
    assert np.allclose(v2[:4], ref[:4], rtol=0, atol=1e-13)
    assert np.all(v2[4] == 0)


def test_xc_flow_fp64_matches_nr_rks(mol, env):
    mf, dm = env
    ni = mf._numint
    ctx = _ctx(mol, 'float64')
    n, e, v = gpu_nr_rks(ctx, ni, mol, mf.grids, 'wb97m-v', dm)
    n0, e0, v0 = ni.nr_rks(mol, mf.grids, 'wb97m-v', dm)
    assert abs(n - n0) < 1e-9
    assert abs(e - e0) < 1e-9
    assert np.max(np.abs(v - v0)) < 1e-9


def test_xc_flow_fp32_within_layer1_gate(mol, env):
    """spec section 8 layer 1: XC energy and Vxc on the same grids."""
    mf, dm = env
    ni = mf._numint
    ctx = _ctx(mol, 'float32')
    n, e, v = gpu_nr_rks(ctx, ni, mol, mf.grids, 'wb97m-v', dm)
    n0, e0, v0 = ni.nr_rks(mol, mf.grids, 'wb97m-v', dm)
    assert abs(e - e0) < GATE_E_FP32
    assert np.max(np.abs(v - v0)) < 1e-5          # potential, looser (FP32)
    assert abs(n - n0) / n0 < 1e-6

@pytest.mark.slow
def test_vv10_flow_fp64_matches_nr_nlc_vxc(mol, env):
    mf, dm = env
    ni = mf._numint
    ctx = _ctx(mol, 'float64')
    from pyscf_wb97mv_fast.gpu.vv10 import gpu_nr_nlc_vxc
    n, e, v = gpu_nr_nlc_vxc(ctx, ni, mol, mf.nlcgrids, 'wb97m-v', dm)
    n0, e0, v0 = ni.nr_nlc_vxc(mol, mf.nlcgrids, 'wb97m-v', dm)
    assert abs(n - n0) < 1e-9
    assert abs(e - e0) < 1e-9
    assert np.max(np.abs(v - v0)) < 1e-9


@pytest.mark.slow
def test_vv10_flow_fp32_within_layer1_gate(mol, env):
    mf, dm = env
    ni = mf._numint
    ctx = _ctx(mol, 'float32')
    from pyscf_wb97mv_fast.gpu.vv10 import gpu_nr_nlc_vxc
    n, e, v = gpu_nr_nlc_vxc(ctx, ni, mol, mf.nlcgrids, 'wb97m-v', dm)
    n0, e0, v0 = ni.nr_nlc_vxc(mol, mf.nlcgrids, 'wb97m-v', dm)
    assert abs(e - e0) < GATE_E
    assert np.max(np.abs(v - v0)) < 1e-5
    # spec section 8 layer 1: rho relative error <= 1e-6 (FP32 GEMM noise
    # accumulates over ~62k grid points, so the gate is on nelec relatively)
    assert abs(n - n0) / n0 < 1e-6


def test_kernel_sum_matches_bruteforce(mol, env):
    """kernel_sum vs a literal double loop of _vv10nlc's formula."""
    mf, dm = env
    ni = mf._numint
    g = mf.nlcgrids
    # small point set so the brute force is cheap
    rng = np.random.default_rng(11)
    sel = rng.choice(g.coords.shape[0], size=140, replace=False)
    sel.sort()
    coords = np.ascontiguousarray(g.coords[sel])
    # per-point rho/grad from the CPU numint
    ao = ni.eval_ao(mol, coords, deriv=1)
    c0 = ao[0] @ dm
    rho0 = (c0 * ao[0]).sum(axis=1)
    grad = 2.0 * np.einsum('pm,gpm->gp', c0, ao[1:4])
    keep = rho0 >= 1e-8
    coords, rho0, grad = coords[keep], rho0[keep], grad[:, keep]
    w = g.weights[sel][keep]

    q, W0, kappa, *_ = vv10_pointwise(
        np.asarray(rho0), np.asarray(grad), np.asarray(w), (5.9, 10.0), np)
    kern = Vv10Kernel(np, mem_budget=1 << 24, tile_tol=0.0, outer_chunk=37)
    F, U, W = kern.kernel_sum(np.asarray(coords, dtype=np.float32),
                              q, W0, kappa)

    R2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g_i = R2 * W0[:, None] + kappa[:, None]
    g_p = R2 * W0[None, :] + kappa[None, :]
    gt = g_i + g_p
    T = q[None, :] / (g_i * g_p * gt)          # T_ij = q_j / (g_i gp_j gt)
    F_ref = -1.5 * T.sum(axis=1)               # VXC_vv10nlc: F = -1.5 x raw
    T2 = T * (1.0 / g_i + 1.0 / gt)
    U_ref = T2.sum(axis=1)
    W_ref = (T2 * R2).sum(axis=1)              # C: W += T*(1/g+1/gt)*R2
    scale = max(np.abs(F_ref).max(), 1e-300)
    assert np.abs(F - F_ref).max() / scale < 1e-5
    assert np.abs(W - W_ref).max() / scale < 1e-5


@pytest.mark.slow
def test_tile_tol_only_removes_negligible_pairs(mol, env):
    mf, dm = env
    ni = mf._numint
    ctx0 = build_context(mol, backend='numpy', blksize=112, verify_ao=False,
                         vv10_tile_tol=0.0)
    ctx1 = build_context(mol, backend='numpy', blksize=112, verify_ao=False,
                         vv10_tile_tol=1e-9)
    from pyscf_wb97mv_fast.gpu.vv10 import gpu_nr_nlc_vxc
    e0 = gpu_nr_nlc_vxc(ctx0, ni, mol, mf.nlcgrids, 'wb97m-v', dm)[1]
    e1 = gpu_nr_nlc_vxc(ctx1, ni, mol, mf.nlcgrids, 'wb97m-v', dm)[1]
    kept, total = ctx1.kernel.last_pairs_kept
    # on a small molecule every pair is above the (loose) bound even at
    # 1e-9: nothing is pruned and the energy is the exact one
    assert kept == total
    assert e1 == e0

    # the pruning decision itself, on crafted tile stats: a far pair is
    # dropped, the diagonal pair is kept
    kern = Vv10Kernel(np, tile_tol=1e-6)
    far = 100.0
    stats = {'q_sum': np.array([1e-3, 1e-3]),
             'w0_min': np.array([1.0, 1.0]),
             'k_min': np.array([1.0, 1.0]),
             'cent': np.array([[0.0, 0.0, 0.0], [far, 0.0, 0.0]]),
             'radius': np.array([1.0, 1.0])}
    mask = kern._pair_mask(stats, stats)
    # diagonal pairs (distance 0) always kept, the far pair dropped
    assert mask[0, 0] and mask[1, 1] and not mask[0, 1] and not mask[1, 0]
    assert kern.last_pairs_kept == (2, 4)


def test_kahan_sum_axis_matches_fsum():
    import math
    rng = np.random.default_rng(5)
    arr = rng.normal(size=(7, 100)).astype(np.float32) * 1e4
    out = kahan_sum_axis(np, arr, axis=1, n_seg=8)
    ref = np.array([math.fsum(row) for row in arr], dtype=np.float32)
    assert np.max(np.abs(out - ref)) / np.abs(ref).max() < 1e-6
    # odd sizes and zero-length axes
    assert kahan_sum_axis(np, arr[:, :13], 1, 8).shape == (7,)
    assert kahan_sum_axis(np, np.zeros((3, 0)), 1, 8).shape == (3,)


def test_kahan_step_compensates():
    big = np.float32(1e8)
    small = np.float32(1.0)
    plain = big
    acc, comp = big, np.float32(0.0)
    for _ in range(1000):
        plain = np.float32(plain + small)
        acc, comp = kahan_step(acc, comp, small)
    # exact target is big + 1000; plain FP32 loses every +1 (ulp(1e8) = 8)
    assert abs((acc + comp) - (big + 1000)) < abs(plain - (big + 1000))


def test_ao_indices_and_scatter(mol):
    pack = ShellPack(mol)
    shells = np.array([1, 3])
    idx = device.ao_indices(pack.ao_loc, shells)
    assert idx[0] == pack.ao_loc[1]
    assert idx.size == pack.ao_loc[2] - pack.ao_loc[1] + pack.ao_loc[4] - pack.ao_loc[3]
    mat = np.zeros((pack.nao, pack.nao))
    blk = np.arange(idx.size ** 2, dtype=float).reshape(idx.size, idx.size)
    device.scatter_add(np, mat, idx, blk)
    assert np.array_equal(mat[np.ix_(idx, idx)], blk)
    device.scatter_add(np, mat, idx, blk)      # second add accumulates
    assert np.array_equal(mat[np.ix_(idx, idx)], 2 * blk)


@pytest.mark.gpu
@pytest.mark.slow
def test_cupy_backend_matches_numpy(mol, env):
    """Layer-1 on the real device: cupy flow vs the CPU numint reference."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    mf, dm = env
    ni = mf._numint
    ctx = build_context(mol, backend='cupy', blksize=112, verify_ao=True)
    n, e, v = gpu_nr_rks(ctx, ni, mol, mf.grids, 'wb97m-v', dm)
    n0, e0, v0 = ni.nr_rks(mol, mf.grids, 'wb97m-v', dm)
    assert abs(e - e0) < GATE_E_FP32
    assert np.max(np.abs(v - v0)) < 1e-5
    from pyscf_wb97mv_fast.gpu.vv10 import gpu_nr_nlc_vxc
    n2, e2, v2 = gpu_nr_nlc_vxc(ctx, ni, mol, mf.nlcgrids, 'wb97m-v', dm)
    n0, e0, v0 = ni.nr_nlc_vxc(mol, mf.nlcgrids, 'wb97m-v', dm)
    assert abs(e2 - e0) < GATE_E_FP32
    assert np.max(np.abs(v2 - v0)) < 1e-5


def _real_vv10_points(mol, mf, dm, npts, seed=3):
    """npts real VV10-grid points (threshed) with their FP64 q, W0, kappa."""
    ni = mf._numint
    g = mf.nlcgrids
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(g.coords.shape[0], size=3 * npts, replace=False))
    coords = g.coords[sel]
    ao = ni.eval_ao(mol, coords, deriv=1)
    c0 = ao[0] @ dm
    rho0 = (c0 * ao[0]).sum(axis=1)
    grad = 2.0 * np.einsum('pm,gpm->gp', c0, ao[1:4])
    keep = np.flatnonzero(rho0 >= 1e-8)[:npts]
    q, W0, kappa = vv10_pointwise(rho0[keep], grad[:, keep], g.weights[sel][keep],
                                  ni.nlc_coeff('wb97m-v')[0][0], np)[:3]
    return coords[keep] - mol.atom_coords().mean(axis=0), q, W0, kappa


def _bruteforce_fuw(coords, q, W0, kappa):
    """FP64 double loop of libdft.VXC_vv10nlc's formula (F = -1.5 x raw)."""
    R2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    g = R2 * W0[:, None] + kappa[:, None]
    gp = R2 * W0[None, :] + kappa[None, :]
    gt = g + gp
    T = q[None, :] / (g * gp * gt)
    Tu = T * (1.0 / g + 1.0 / gt)
    return -1.5 * T.sum(axis=1), Tu.sum(axis=1), (Tu * R2).sum(axis=1)


@pytest.mark.gpu
def test_kernel_sum_cupy_matches_fp64_bruteforce(mol, env):
    """spec section 8 layer 1: kernel_sum F, U, W relative error <= 1e-6, on
    the device path.  1100 points: several 256-point tiles plus a ragged
    tail; all three outputs are checked (the numpy test above skips U)."""
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    mf, dm = env
    coords, q, W0, kappa = _real_vv10_points(mol, mf, dm, 1100)
    assert coords.shape[0] == 1100
    kern = Vv10Kernel(cupy, tile_tol=0.0)
    out = kern.kernel_sum(cupy.asarray(coords, dtype=cupy.float32), cupy.asarray(q),
                          cupy.asarray(W0), cupy.asarray(kappa))
    for got, ref, name in zip(out, _bruteforce_fuw(coords, q, W0, kappa), 'FUW'):
        got = cupy.asnumpy(got)
        assert got.dtype == np.float64, name
        rel = np.abs(got - ref) / np.abs(ref)
        assert rel.max() < 1e-6, (name, rel.max())


@pytest.mark.gpu
def test_kernel_sum_cupy_throughput_floor():
    """Performance floor on the slowest supported card (2080 Ti): the old
    elementwise-chain kernel_sum ran 2.8e9 pairs/s, 4x slower than the whole
    CPU nr_nlc_vxc on water27 (benchmarks/results/profile_gpu_cycle_water27.json).
    A fused kernel must clear 1e10 pairs/s."""
    import time
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    rng = np.random.default_rng(0)
    n = 65536
    args = [cupy.asarray(rng.uniform(-10, 10, (n, 3)), dtype=cupy.float32),
            cupy.asarray(rng.uniform(1e-4, 1e-2, n)), cupy.asarray(rng.uniform(0.5, 3, n)),
            cupy.asarray(rng.uniform(0.5, 2, n))]
    kern = Vv10Kernel(cupy, tile_tol=0.0)
    kern.kernel_sum(*args)
    cupy.cuda.Device().synchronize()
    t0 = time.perf_counter()
    kern.kernel_sum(*args)
    cupy.cuda.Device().synchronize()
    rate = n * n / (time.perf_counter() - t0)
    assert rate > 1e10, '%.3e pairs/s' % rate


def test_tile_pruning_uses_the_inner_tile_of_each_pair():
    """tile_tol > 0 path: the pair mask must index (outer tile, inner tile)
    with the tile sizes actually used in the loop.  Two clusters 1000 bohr
    apart, outer tiles of 256 points, inner tiles of 1024: the far pairs are
    negligible, so pruned and exact sums must agree.  The old mask was built
    on outer-size tiles for both axes, which dropped cluster B's own inner
    tile for B's outer tiles."""
    rng = np.random.default_rng(7)
    n = 1024
    a = rng.uniform(-2, 2, (n, 3))
    b = rng.uniform(-2, 2, (n, 3)) + np.array([1000.0, 0.0, 0.0])
    coords = np.concatenate([a, b]).astype(np.float32)
    q = rng.uniform(1e-4, 1e-3, 2 * n)
    W0 = rng.uniform(0.5, 2.0, 2 * n)
    kappa = rng.uniform(0.5, 2.0, 2 * n)
    exact = Vv10Kernel(np, mem_budget=1, tile_tol=0.0, outer_chunk=256)
    pruned = Vv10Kernel(np, mem_budget=1, tile_tol=1e-12, outer_chunk=256)
    assert exact._inner_chunk() == 1024           # tiles differ on the two axes
    ref = exact.kernel_sum(coords, q, W0, kappa)
    got = pruned.kernel_sum(coords, q, W0, kappa)
    kept, total = pruned.last_pairs_kept
    assert kept < total                           # something was pruned
    for g_, r_, name in zip(got, ref, 'FUW'):
        assert np.max(np.abs(g_ - r_) / np.abs(r_)) < 1e-6, name


def test_xc_flow_fp32_has_no_constant_rounding_bias(mol, env):
    """FP32 AO evaluation must not carry the basis constants' FP32 rounding
    (c2s normalization, contraction coefficients, exponents, centers): that
    rounding is the same for every grid point, so it biases nelec and E_xc
    and grows with system size (benchmarks/xc/FINDINGS.md: dimer -3.1e-6,
    water27 -2.7e-5 Ha).  With the constants compensated, the FP32 flow on
    the dimer must land near 'AO exact, the rest FP32' (~2e-7).  Measured on
    this dm: dN 2.3e-6 before (constants rounded, CUDA expf); with the
    constants compensated and exp in FP64 the residual is ~5e-7 (a further
    FP32-arithmetic bias, not yet located -- benchmarks/xc/FINDINGS.md)."""
    mf, dm = env
    ni = mf._numint
    n32, e32, _ = gpu_nr_rks(_ctx(mol, 'float32'), ni, mol, mf.grids, 'wb97m-v', dm)
    n64, e64, _ = gpu_nr_rks(_ctx(mol, 'float64'), ni, mol, mf.grids, 'wb97m-v', dm)
    assert abs(n32 - n64) < 1e-6, n32 - n64
    assert abs(e32 - e64) < 1e-6, e32 - e64
