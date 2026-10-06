"""S5 Tasks 4+5: fused GPU COSX K vs sgx_jk.get_k_only, and screening."""
import numpy as np
import pytest
from pyscf import dft
from pyscf.sgx import sgx_jk

from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends


def _sgx(system, level=2, basis='def2-svp'):
    mol = build_mol(system, basis)
    mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
    sgx = mf.with_df
    sgx.grids_level_i = sgx.grids_level_f = level
    sgx.build(level=level)
    dm = mf.get_init_guess()
    dm = (dm + dm.T) / 2
    return mol, mf, sgx, dm


@pytest.mark.gpu
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_fused_k_matches_cpu_get_k_only(omega):
    """Energy gate: long-range K 1e-7; full K 3e-7 = the FP32 coherent
    rounding floor on the atom-centred SGX grid (2.28e-7 / 1.88e-7 measured)
    plus margin -- Ruling A, 2026-10-06 (docs/2026-10-03-decisions.md).  The
    FP32 K only drives the FP32 stages; the final energy comes from the FP64
    tail, which uses the CPU K."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
    mol, mf, sgx, dm = _sgx('water_dimer')
    with mol.with_range_coulomb(omega):
        k_cpu = sgx_jk.get_k_only(sgx, dm, hermi=1)
        b = GpuKBuilder(mol, cp, tile_tol=0.0)
        k_gpu = get_k_only_gpu(b, sgx, dm, hermi=1)
        k_gpu2 = get_k_only_gpu(b, sgx, dm, hermi=1)
    assert k_gpu.dtype == np.float64
    assert np.array_equal(k_gpu, k_gpu2)                     # reproducible
    assert np.max(np.abs(k_gpu - k_cpu)) < 1e-6 * np.abs(k_cpu).max()
    gate = 3e-7 if omega == 0.0 else 1e-7                    # Ruling A
    assert abs(0.25 * np.einsum('ij,ji->', dm, k_gpu - k_cpu)) < gate
    assert np.allclose(k_gpu, k_gpu.T, rtol=0, atol=0)       # hermi=1 exact symmetry


@pytest.mark.gpu
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_fused_k_tzvp_matches_cpu_get_k_only(omega):
    """f shells (def2-tzvp O): the same matrix criterion as def2-svp; the
    energy error is only reported (energy-level TZVP check: six drug
    molecules vs stock PySCF)."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
    mol, mf, sgx, dm = _sgx('water_dimer', basis='def2-tzvp')
    with mol.with_range_coulomb(omega):
        k_cpu = sgx_jk.get_k_only(sgx, dm, hermi=1)
        b = GpuKBuilder(mol, cp, tile_tol=0.0)
        k_gpu = get_k_only_gpu(b, sgx, dm, hermi=1)
        k_gpu2 = get_k_only_gpu(b, sgx, dm, hermi=1)
    assert np.array_equal(k_gpu, k_gpu2)
    assert np.max(np.abs(k_gpu - k_cpu)) < 1e-6 * np.abs(k_cpu).max()
    assert np.allclose(k_gpu, k_gpu.T, rtol=0, atol=0)
    print('tzvp dimer omega=%g: dE_K = %.3e' % (
        omega, 0.25 * np.einsum('ij,ji->', dm, k_gpu - k_cpu)))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_fused_k_water27_energy_error(omega):
    """FP32-stage budget on water27 (the FP64 tail uses the CPU K, so this
    only shapes the convergence path): |dE_K| < 2e-6 for the long-range K,
    3e-6 for the full K = FP32 coherent rounding floor (2.24e-6 on the 4090,
    2.76e-6 on the 2080 Ti) plus margin, Ruling A 2026-10-06."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
    mol, mf, sgx, _ = _sgx('water27')
    dm = np.load('benchmarks/experimental/cosx/dm_water27_def2-svp.npy')
    dm = (dm + dm.T) / 2
    with mol.with_range_coulomb(omega):
        k_cpu = sgx_jk.get_k_only(sgx, dm, hermi=1)
        k_gpu = get_k_only_gpu(GpuKBuilder(mol, cp, tile_tol=0.0), sgx, dm,
                               hermi=1)
    assert np.linalg.norm(k_gpu - k_cpu) / np.linalg.norm(k_cpu) < 1e-5
    gate = 3e-6 if omega == 0.0 else 2e-6                    # Ruling A
    assert abs(0.25 * np.einsum('ij,ji->', dm, k_gpu - k_cpu)) < gate


# -- Task 5: block screening ---------------------------------------------------

@pytest.mark.gpu
def test_tile_tol_zero_is_bitwise_unscreened():
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
    mol, mf, sgx, dm = _sgx('water_dimer')
    b = GpuKBuilder(mol, cp, tile_tol=0.0)
    k1 = get_k_only_gpu(b, sgx, dm, hermi=1)
    assert b.last_stats['pairs_kept'] == b.last_stats['pairs_total']
    k2 = get_k_only_gpu(GpuKBuilder(mol, cp, tile_tol=0.0), sgx, dm, hermi=1)
    assert np.array_equal(k1, k2)


@pytest.mark.gpu
@pytest.mark.parametrize('basis,lmax', [('def2-svp', 2), ('def2-tzvp', 3)])
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_specialized_kernels_match_generic(omega, basis, lmax):
    """Each (la, lb)-specialized kernel on its own task list must give the
    G of the generic kernel on the same list: same math, only compile-time
    sizes and the host-side theta p (water dimer: s, p, d shells in def2-svp
    -> 9 classes; plus f on O in def2-tzvp -> all 16).

    Judged against an FP64 G on 500 sampled points (PySCF int1e_grids,
    contracted in the kernel's convention): |spec - gen| may not exceed
    max(1e-6 * class max, the generic kernel's own FP64 error).  Some classes
    are ill-conditioned in FP32 -- (3,1) at omega 0.3: both kernels 9.3e-5
    off FP64 and 5.8e-6 apart (def2-TZVP)
    -- so a fixed 1e-6 flags rounding, while a wrong index or term in the
    specialized code still differs by the class max itself."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, GENERIC, _nsub
    from pyscf_wb97mv_fast.gpu.shellpairs import LMAX
    mol, mf, sgx, dm = _sgx('water_dimer', basis=basis)
    b = GpuKBuilder(mol, cp, tile_tol=0.0)
    b._build()
    st = b._grid_state(sgx.grids)
    i0, i1 = st.tiles[0]
    sgx._build_pjs(1e-13)
    pdm = cp.asarray(sgx._pjs_data._overlap_correction_matrix @ dm)
    X = b._ao.eval(st.coords64[i0:i1], np.arange(mol.nbas), deriv=0)[0]
    F = (X @ pdm).astype(cp.float32)
    nsub = _nsub(i1 - i0, st.ks)
    tasks, _ = b._tasks(st, i0, i1, F, nsub)
    assert sorted(c for c, _, _ in tasks) == [la * (LMAX + 1) + lb
                                              for la in range(lmax + 1)
                                              for lb in range(lmax + 1)]
    rows = np.sort(np.random.default_rng(5).choice(i1 - i0, 500, replace=False))
    coords = cp.asnumpy(st.coords64[i0:i1])[rows] + b.origin   # kernel frame -> lab
    with mol.with_range_coulomb(omega):
        A = mol.intor('int1e_grids', grids=coords)              # (500, nao, nao) FP64
    F64 = cp.asnumpy(F)[rows].astype(np.float64)
    pair_rows, sh_dims = b._pairs_host.pairs.pair_rows, b._pairs_host.pairs.sh_dims
    for cls, offs, ids in tasks:
        g_spec = cp.asnumpy(b._launch(st, i0, i1, F, [(cls, offs, ids)], nsub, omega))[rows]
        g_gen = cp.asnumpy(b._launch(st, i0, i1, F, [(GENERIC, offs, ids)], nsub, omega))[rows]
        g_ref = np.zeros(g_gen.shape)
        for k in np.unique(cp.asnumpy(ids)):    # G[g,nu] += A F[mu] (+ transposed)
            ia0, ib0, ish, jsh = (int(x) for x in pair_rows[k][:4])
            mu = slice(ia0, ia0 + int(sh_dims[ish][0]))
            nu = slice(ib0, ib0 + int(sh_dims[jsh][0]))
            g_ref[:, nu] += np.einsum('gmn,gm->gn', A[:, mu, nu], F64[:, mu])
            if ia0 != ib0:
                g_ref[:, mu] += np.einsum('gmn,gn->gm', A[:, mu, nu], F64[:, nu])
        scale = np.abs(g_ref).max()
        d_sg = np.max(np.abs(g_spec - g_gen))
        e_gen = np.max(np.abs(g_gen - g_ref))
        assert d_sg <= max(1e-6 * scale, e_gen), (cls, d_sg / scale, e_gen / scale)


@pytest.mark.gpu
@pytest.mark.parametrize('tol', [1e-11, 1e-7, 1e-5])
def test_device_screening_keeps_superset_of_host_reference(tol):
    """The device keep mask (P4, FP32 with a conservative margin) against
    the host FP64 reference (_kept_host: the locked pair_bound per
    sub-block x the F estimate), compared as task sets per tile: every
    task the exact bound keeps must be kept on the device (the device
    estimate is a bound of the bound), and the extra ones -- estimates
    within ~0.1% of tile_tol -- must be rare."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, _nsub
    mol, mf, sgx, dm = _sgx('water_dimer')
    b = GpuKBuilder(mol, cp, tile_tol=tol)
    b._build()
    st = b._grid_state(sgx.grids)
    sgx._build_pjs(1e-13)
    pdm = cp.asarray(sgx._pjs_data._overlap_correction_matrix @ dm)
    npairs = b._pairs_host.npairs

    def task_set(tasks, nsub):
        out = set()
        for _, offs, ids in tasks:
            offs = cp.asnumpy(offs); ids = cp.asnumpy(ids)
            sub = np.repeat(np.arange(nsub), np.diff(offs))
            out.update((sub * npairs + ids).tolist())
        return out

    total = extra = 0
    for (i0, i1) in st.tiles:
        X = b._ao.eval(st.coords64[i0:i1], np.arange(mol.nbas), deriv=0)[0]
        F = (X @ pdm).astype(cp.float32)
        nsub = _nsub(i1 - i0, st.ks)
        b.host_screening = True
        host = task_set(b._tasks(st, i0, i1, F, nsub)[0], nsub)
        b.host_screening = False
        dev = task_set(b._tasks(st, i0, i1, F, nsub)[0], nsub)
        assert host <= dev, (tol, len(host - dev))
        extra += len(dev - host)
        total += nsub * npairs
    assert extra <= max(2, 1e-3 * total), (tol, extra, total)


@pytest.mark.gpu
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_launch_shape_does_not_change_k(omega):
    """The warps per CUDA block are picked per device (2080 Ti / 5080 /
    5090 differ): every thread owns its G row and accumulates it in a fixed
    order, so the launch shape must not change a single bit of K."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
    mol, mf, sgx, dm = _sgx('water_dimer')
    with mol.with_range_coulomb(omega):
        ks = [get_k_only_gpu(GpuKBuilder(mol, cp, tile_tol=1e-11, warps_per_block=w),
                             sgx, dm, hermi=1) for w in (1, 3, None)]
    assert np.array_equal(ks[0], ks[1]) and np.array_equal(ks[0], ks[2])


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_screening_error_and_savings_water27(omega):
    """water27, SGX level 2, dm from file: screened vs unscreened
    |0.25 tr(D dK)| <= 1e-8 Ha and max|dK| <= 1e-8, and at least 50% of the
    (pair, block) tasks skipped (CPU pjs relies on the same F = P X decay)."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
    mol, mf, sgx, _ = _sgx('water27')
    dm = np.load('benchmarks/experimental/cosx/dm_water27_def2-svp.npy')
    dm = (dm + dm.T) / 2
    with mol.with_range_coulomb(omega):
        k0 = get_k_only_gpu(GpuKBuilder(mol, cp, tile_tol=0.0), sgx, dm,
                            hermi=1)
        bs = GpuKBuilder(mol, cp)                    # default tile_tol
        ks = get_k_only_gpu(bs, sgx, dm, hermi=1)
    assert np.max(np.abs(ks - k0)) < 1e-8
    assert abs(0.25 * np.einsum('ij,ji->', dm, ks - k0)) < 1e-8
    assert bs.last_stats['pairs_kept'] <= 0.5 * bs.last_stats['pairs_total']


@pytest.mark.gpu
def test_out_of_scope_raises_unsupported():
    """Multiple dms / hermi=0 / non-cupy backend raise Unsupported (the
    install hook falls back to the CPU with a warning; nothing silent)."""
    from pyscf_wb97mv_fast.gpu.sgx_k import GpuKBuilder, get_k_only_gpu
    from pyscf_wb97mv_fast.gpu.xc import Unsupported
    mol, mf, sgx, dm = _sgx('water_dimer')
    b = GpuKBuilder(mol, backends.get_xp('numpy'))
    with pytest.raises(Unsupported):
        get_k_only_gpu(b, sgx, dm, hermi=1)
    cp = backends.cupy_module()
    if cp is None:
        pytest.skip('no CUDA device for the hermi/multi-dm checks')
    b = GpuKBuilder(mol, cp, tile_tol=0.0)
    with pytest.raises(Unsupported):
        get_k_only_gpu(b, sgx, dm, hermi=0)
    with pytest.raises(Unsupported):
        get_k_only_gpu(b, sgx, np.stack([dm, 0.5 * dm]), hermi=1)
