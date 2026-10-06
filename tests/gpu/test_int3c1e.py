"""S5 Task 3: dense FP32 int3c1e kernel vs the FP64 NumPy reference."""
import numpy as np
import pytest

from pyscf_wb97mv_fast.core.testsystems import build_mol
from pyscf_wb97mv_fast.gpu import backends
from pyscf_wb97mv_fast.gpu.shellpairs import build_shell_pairs


def _points(mol, rng, n=200):
    span = np.abs(mol.atom_coords()).max() + 3.0
    pts = rng.uniform(-span, span, (n, 3))
    far = np.array([[100.0, 0.0, 0.0], [0.0, -80.0, 30.0]])
    return np.vstack([mol.atom_coords(), far, pts])   # nuclei + far points


@pytest.mark.gpu
@pytest.mark.parametrize('basis', ['def2-svp', 'def2-tzvp'])  # O: up to d / f
@pytest.mark.parametrize('omega', [0.0, 0.3])
def test_fp32_kernel_matches_fp64_reference(omega, basis):
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
    mol = build_mol('water_dimer', basis)
    pts = _points(mol, np.random.default_rng(3))       # same helper as Task 2
    pairs = build_shell_pairs(mol)
    # PySCF's C int1e_grids: int3c1e_reference equals it to 1e-12
    # (test_shellpairs.py::test_reference_matches_pyscf_int1e_grids), and it
    # takes milliseconds where the NumPy reference takes about a minute
    with mol.with_range_coulomb(omega):
        ref = mol.intor('int1e_grids', grids=pts)
    got = cp.asnumpy(int3c1e_fp32(cp, pairs.to_device(cp),
                                  cp.asarray(pts), omega=omega))
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - ref)) < 1e-6 * np.abs(ref).max()
    # no systematic bias: mean signed error on the large entries << eps
    big = np.abs(ref) > 1e-3 * np.abs(ref).max()
    assert abs(np.mean((got[big] - ref[big]) / ref[big])) < 1e-8


@pytest.mark.gpu
@pytest.mark.parametrize('basis', ['def2-svp', 'def2-tzvp'])
def test_fp32_kernel_is_symmetric_and_reproducible(basis):
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
    mol = build_mol('water_dimer', basis)
    pts = _points(mol, np.random.default_rng(4))
    pairs = build_shell_pairs(mol)
    dev = pairs.to_device(cp)
    got = cp.asnumpy(int3c1e_fp32(cp, dev, cp.asarray(pts)))
    assert np.array_equal(got, got.transpose(0, 2, 1))
    got2 = cp.asnumpy(int3c1e_fp32(cp, dev, cp.asarray(pts)))
    assert np.array_equal(got, got2)                   # bitwise reproducible


@pytest.mark.gpu
def test_fp32_kernel_refuses_oversized_tensor():
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.int3c1e import int3c1e_fp32
    mol = build_mol('water_dimer', 'def2-svp')
    pairs = build_shell_pairs(mol)
    dev = pairs.to_device(cp, mem_budget=1024)         # absurdly small budget
    with pytest.raises(MemoryError):
        int3c1e_fp32(cp, dev, cp.zeros((8, 3), dtype=cp.float32))
