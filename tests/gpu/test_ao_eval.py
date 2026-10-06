"""S3: packed-basis AO evaluation (gpu.ao_eval).

These run on the numpy backend -- the same code path the GPU takes -- so
they need no CUDA device.  The gpu-marked cross-checks live in
test_install.py.
"""
import numpy as np
import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends
from pyscf_wb97mv_fast.gpu.ao_eval import (AoEvaluator, ShellPack,
                                           bucketize, cart_components,
                                           plan_blocks)


@pytest.fixture(scope='module')
def mol():
    return build_mol('water_dimer', 'def2-svp')


@pytest.fixture(scope='module')
def grids(mol):
    g = dft.gen_grid.Grids(mol)
    g.level = 2
    g.build(with_non0tab=True)
    return g


@pytest.fixture(scope='module')
def evaluator(mol):
    # FP64: this file pins conventions (values vs mol.eval_gto at 1e-9);
    # FP32 accuracy is covered by the layer-1 gates in test_xc_vv10.py
    return AoEvaluator(mol, backends.get_xp('numpy'), verify=False,
                       dtype='float64')


def test_self_check_passes(mol):
    """The pack-time self-check (vs mol.eval_gto) accepts the conventions."""
    AoEvaluator(mol, backends.get_xp('numpy'), verify=True)  # raises on mismatch


def test_cart_component_order_is_libcint(mol):
    # lx desc, ly desc, lz = l-lx-ly; matches numint.eval_ao's docstring order
    assert cart_components(2) == [(2, 0, 0), (1, 1, 0), (1, 0, 1),
                                  (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    assert cart_components(3)[:4] == [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0)]


def test_ao_values_and_gradients_match_eval_gto(mol, evaluator):
    rng = np.random.default_rng(7)
    pts = mol.atom_coords().mean(axis=0) + rng.uniform(-4, 4, size=(64, 3))
    shells = np.arange(mol.nbas)
    mine0 = evaluator.eval(pts - evaluator.pack.origin, shells, deriv=0)
    mine1 = evaluator.eval(pts - evaluator.pack.origin, shells, deriv=1)
    ref0 = mol.eval_gto('GTOval_sph', pts, comp=1)
    ref1 = mol.eval_gto('GTOval_sph_deriv1', pts)
    assert mine0.shape == (1, 64, mol.nao) and mine1.shape == (4, 64, mol.nao)
    assert np.allclose(mine0[0], ref0, rtol=0, atol=1e-9)
    assert np.allclose(mine1, ref1, rtol=0, atol=1e-9)


def test_active_shell_columns_equal_full_eval(mol, grids, evaluator):
    """AO columns of the active shell set must equal the same columns of a
    full-shell evaluation (the inactive ones are what libcint zeroes)."""
    blocks = plan_blocks(mol, grids, blksize=112)
    i0, i1, shells = blocks[0]
    coords = grids.coords[i0:i1] - evaluator.pack.origin
    sub = evaluator.eval(coords, shells, deriv=0)[0]
    full = evaluator.eval(coords, np.arange(mol.nbas), deriv=0)[0]
    pos = np.zeros(len(shells) + 1, dtype=int)
    widths = evaluator.pack.ao_loc[shells + 1] - evaluator.pack.ao_loc[shells]
    pos[1:] = np.cumsum(widths)
    for k, sh in enumerate(shells):
        lo_full = evaluator.pack.ao_loc[sh]
        hi_full = evaluator.pack.ao_loc[sh + 1]
        assert np.array_equal(sub[:, pos[k]:pos[k + 1]],
                              full[:, lo_full:hi_full])


def test_active_set_matches_non0tab(mol, grids, evaluator):
    """A shell excluded by non0tab on a sub-block contributes nothing there:
    libcint zeroes it, so the active-set sum equals the full dense sum."""
    blocks = plan_blocks(mol, grids, blksize=112)
    i0, i1, shells = blocks[0]
    coords = grids.coords[i0:i1] - evaluator.pack.origin
    sub = evaluator.eval(coords, shells, deriv=0)[0]
    full = evaluator.eval(coords, np.arange(mol.nbas), deriv=0)[0]
    width_sub = int((evaluator.pack.ao_loc[shells + 1]
                     - evaluator.pack.ao_loc[shells]).sum())
    assert sub.shape[1] == width_sub
    # rebuild the full-width matrix from the active columns
    rebuilt = np.zeros_like(full)
    pos = np.zeros(len(shells) + 1, dtype=int)
    widths = evaluator.pack.ao_loc[shells + 1] - evaluator.pack.ao_loc[shells]
    pos[1:] = np.cumsum(widths)
    for k, sh in enumerate(shells):
        rebuilt[:, evaluator.pack.ao_loc[sh]:evaluator.pack.ao_loc[sh + 1]] = \
            sub[:, pos[k]:pos[k + 1]]
    # shells the mask dropped must be negligible on this block; PySCF's
    # grid screening leaves value tails up to ~1e-4 (gen_grid screening)
    dropped = np.setdiff1d(np.arange(mol.nbas), shells)
    if dropped.size:
        assert np.abs(full[:, dropped]).max() < 1e-3
    assert np.allclose(rebuilt, full, rtol=0, atol=1e-12)


def test_plan_blocks_covers_grid(mol, grids):
    blocks = plan_blocks(mol, grids, blksize=112)
    ngrids = grids.coords.shape[0]
    covered = sum(i1 - i0 for i0, i1, _ in blocks)
    assert covered == ngrids
    assert blocks[0][0] == 0 and blocks[-1][1] == ngrids
    for i0, i1, shells in blocks[:-1]:
        assert (i1 - i0) % 56 == 0
        assert np.all(np.diff(shells) > 0)
    # all shells present when non0tab is unavailable
    g2 = dft.gen_grid.Grids(mol)
    g2.level = 1
    g2.build(with_non0tab=False)
    b2 = plan_blocks(mol, g2, blksize=56)
    assert len(b2[0][2]) == mol.nbas


def test_bucketize_groups_by_shape(mol, grids):
    blocks = plan_blocks(mol, grids, blksize=224)
    buckets = bucketize(blocks)
    assert sum(len(v) for v in buckets.values()) == len(blocks)
    assert list(buckets) == sorted(buckets)            # deterministic order


def test_shellpack_rejects_kappa(mol):
    mol2 = build_mol('water_dimer', 'def2-svp')
    mol2.bas_kappa = lambda sh: -abs(int(mol2.bas_angular(sh)))  # spinor-like
    with pytest.raises(ValueError):
        ShellPack(mol2)
