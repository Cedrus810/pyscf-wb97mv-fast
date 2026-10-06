"""S1 Task 5: CPU FP64 reference wrappers around PySCF numint."""
import numpy as np
import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core.hooks import HookSet
from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.reference.numint import vv10_reference, xc_reference


@pytest.fixture(scope='module')
def built():
    mol = build_mol('water_dimer', 'def2-svp')
    mf = dft.RKS(mol, xc='wb97m-v')
    dm = mf.get_init_guess()
    mf.grids.build(with_non0tab=True)
    mf.nlcgrids.build(with_non0tab=True)
    return mf, dm


# PySCF's threaded numint is not bit-reproducible run to run (OpenMP reduction
# order): two identical nr_rks calls differ by ~4e-16 with 80 threads and are
# identical with 1 thread. Multi-threaded comparisons use tolerances far below
# any real difference; the single-thread test pins exact equality.
ATOL_E, ATOL_V = 1e-12, 1e-13


def test_xc_reference_equals_pyscf(built):
    mf, dm = built
    n, e, v = xc_reference(mf, dm)
    n0, e0, v0 = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)
    assert abs(n - n0) < ATOL_E and abs(e - e0) < ATOL_E
    assert np.allclose(v, v0, rtol=0, atol=ATOL_V)


def test_vv10_reference_equals_pyscf(built):
    mf, dm = built
    n, e, v = vv10_reference(mf, dm)
    n0, e0, v0 = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)
    assert e != 0.0
    assert abs(n - n0) < ATOL_E and abs(e - e0) < ATOL_E
    assert np.allclose(v, v0, rtol=0, atol=ATOL_V)


@pytest.mark.slow
def test_references_bitwise_equal_single_thread(built):
    from pyscf import lib
    mf, dm = built
    with lib.with_omp_threads(1):
        v = xc_reference(mf, dm)[2]
        v0 = mf._numint.nr_rks(mf.mol, mf.grids, mf.xc, dm)[2]
        w = vv10_reference(mf, dm)[2]
        w0 = mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)[2]
    assert np.array_equal(v, v0) and np.array_equal(w, w0)


def test_references_bypass_instance_hooks(built):
    mf, dm = built
    e_xc = xc_reference(mf, dm)[1]
    e_nl = vv10_reference(mf, dm)[1]
    zeros = lambda orig: (lambda *a, **k: (0.0, 0.0, np.zeros_like(dm)))  # noqa: E731
    with HookSet() as hooks:
        hooks.wrap(mf._numint, 'nr_rks', zeros)
        hooks.wrap(mf._numint, 'nr_nlc_vxc', zeros)
        assert mf._numint.nr_nlc_vxc(mf.mol, mf.nlcgrids, mf.xc, dm)[1] == 0.0   # hook active
        assert abs(xc_reference(mf, dm)[1] - e_xc) < ATOL_E
        assert abs(vv10_reference(mf, dm)[1] - e_nl) < ATOL_E


def test_unbuilt_grids_are_prepared_like_initialize_grids():
    """Fresh mf: the reference must use the same (small-rho pruned) grids that
    rks.initialize_grids would build for get_veff, not raw unpruned grids."""
    from pyscf_wb97mv_fast.reference.numint import prepared_grids
    mol = build_mol('water_dimer', 'def2-svp')
    mf = dft.RKS(mol, xc='wb97m-v')
    dm = mf.get_init_guess()
    ref = dft.RKS(mol, xc='wb97m-v')
    ref.initialize_grids(mol, dm)
    g = prepared_grids(mf, dm, kind='xc')
    h = prepared_grids(mf, dm, kind='nlc')
    assert g.weights.size == ref.grids.weights.size
    assert h.weights.size == ref.nlcgrids.weights.size
    assert np.array_equal(g.coords, ref.grids.coords)
    assert mf.grids.coords is None and mf.nlcgrids.coords is None      # mf untouched


def test_references_do_not_mutate_mf():
    mol = build_mol('water_dimer', 'def2-svp')
    mf = dft.RKS(mol, xc='wb97m-v')
    dm = mf.get_init_guess()
    level_before = mf.nlcgrids.level
    assert mf.grids.coords is None and mf.nlcgrids.coords is None
    xc_reference(mf, dm)
    vv10_reference(mf, dm)
    assert mf.grids.coords is None and mf.nlcgrids.coords is None
    assert mf.nlcgrids.level == level_before


def test_explicit_grids_argument(built):
    mf, dm = built
    g = dft.gen_grid.Grids(mf.mol)
    g.level = 1
    g.build(with_non0tab=True)
    e_coarse = vv10_reference(mf, dm, grids=g)[1]
    e_default = vv10_reference(mf, dm)[1]
    assert e_coarse != e_default                 # really used the given grid
    assert abs(e_coarse - e_default) < 1e-3      # same functional, other grid
