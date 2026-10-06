"""S5 Task 2: shell-pair preprocessing, FP64 reference integrals and the
screening bound (gpu.shellpairs).  CPU only -- no CUDA device needed."""
import numpy as np
import pytest
from pyscf import gto

from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu.shellpairs import (build_shell_pairs,
                                              int3c1e_reference, pair_bound)
from pyscf_wb97mv_fast.gpu.xc import Unsupported


def _points(mol, rng, n=200):
    span = np.abs(mol.atom_coords()).max() + 3.0
    pts = rng.uniform(-span, span, (n, 3))
    far = np.array([[100.0, 0.0, 0.0], [0.0, -80.0, 30.0]])
    return np.vstack([mol.atom_coords(), far, pts])   # nuclei + far points


@pytest.mark.parametrize('basis', ['def2-svp', 'def2-tzvp'])  # O: up to d / f
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_reference_matches_pyscf_int1e_grids(omega, basis):
    mol = build_mol('water_dimer', basis)
    # nuclei + far points + 24 random: an elementwise 1e-12 check needs no more,
    # and the NumPy reference costs about 0.3 s per point here
    pts = _points(mol, np.random.default_rng(1), n=24)
    pairs = build_shell_pairs(mol)
    mine = int3c1e_reference(pairs, pts, omega=omega)
    with mol.with_range_coulomb(omega):
        ref = mol.intor('int1e_grids', grids=pts)
    assert mine.shape == ref.shape == (len(pts), mol.nao, mol.nao)
    assert np.all(np.isfinite(mine))
    assert np.max(np.abs(mine - ref)) < 1e-12 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize('basis', ['def2-svp', 'def2-tzvp'])
@pytest.mark.parametrize('omega', [0.0, 0.3])
@pytest.mark.slow
def test_bound_is_an_upper_bound(omega, basis):
    """Blocks of random points, blocks that contain the nuclei, tight blocks
    around a single nucleus and far blocks: max |A| <= bound for every pair."""
    mol = build_mol('water_dimer', basis)
    rng = np.random.default_rng(2)
    pairs = build_shell_pairs(mol)
    blocks = [_points(mol, rng, 64)[:64] for _ in range(5)]
    blocks += [mol.atom_coords()[k] + rng.normal(scale=0.05, size=(32, 3))
               for k in range(mol.natm)]
    blocks += [np.array([40.0, 0.0, 0.0]) + rng.normal(scale=1.0, size=(32, 3))]
    ab = pairs.ao_blocks()
    for blk in blocks:
        A = int3c1e_reference(pairs, blk, omega=omega)
        b = pair_bound(pairs, blk)
        assert b.shape == (pairs.npairs,)
        for k, (i0, i1, j0, j1) in enumerate(ab):
            assert np.abs(A[:, i0:i1, j0:j1]).max() <= b[k] * (1 + 1e-12), \
                (k, omega)


def test_bound_monotone_with_coulomb_tail():
    """The integrals only decay like 1/d with the grid distance (Coulomb
    tail: A ~ S_mu,nu / d), so the bound must too -- neither faster (it would
    stop being a bound) nor slower (it would be useless).  Screening power
    comes from F = X proj P, not from A (spec section 8.5)."""
    mol = build_mol('water_dimer', 'def2-svp')
    pairs = build_shell_pairs(mol)
    centre = mol.atom_coords().mean(axis=0)
    blk = lambda d: centre + np.array([d, 0.0, 0.0]) + np.zeros((1, 3))
    b20, b40, b80 = (pair_bound(pairs, blk(d)) for d in (20.0, 40.0, 80.0))
    assert np.all(b40 <= b20) and np.all(b80 <= b40)      # monotone in distance
    ratio = b40 / b80                                     # -> 2 for a 1/d tail
    assert np.all(np.abs(ratio - 2.0) < 0.1), (ratio.min(), ratio.max())


def test_g_functions_unsupported():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59',
                basis='def2-qzvp', verbose=0)                   # g on O
    with pytest.raises(Unsupported):
        build_shell_pairs(mol)


def test_f_functions_supported():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59',
                basis='def2-tzvp', verbose=0)
    pairs = build_shell_pairs(mol)
    assert pairs.sh_l.max() == 3


def test_cartesian_basis_unsupported():
    mol = gto.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59',
                basis='def2-svp', verbose=0, cart=True)
    with pytest.raises(Unsupported):
        build_shell_pairs(mol)


def test_build_is_reproducible_and_sorted():
    """Same input -> identical pair table; pairs ordered by (ish, jsh)."""
    mol = build_mol('water_dimer', 'def2-svp')
    p1 = build_shell_pairs(mol)
    p2 = build_shell_pairs(mol)
    assert np.array_equal(p1.pair_rows, p2.pair_rows)
    assert np.array_equal(p1.prim_p, p2.prim_p)
    order = [(r[2], r[3]) for r in p1.pair_rows]
    assert order == sorted(order)
