import numpy as np
from pyscf_wb97mv_fast.core.testsystems import build_mol


def test_chain_atom_count():
    assert build_mol('chain4', 'sto-3g').natm == 14


def test_water_cube_count_and_no_clash():
    mol = build_mol('water27', 'sto-3g')
    assert mol.natm == 81
    xyz = mol.atom_coords(unit='Angstrom')
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    mol_id = np.repeat(np.arange(27), 3)
    inter = d[mol_id[:, None] != mol_id[None, :]]
    assert inter.min() > 1.6


def test_water_dimer():
    assert build_mol('water_dimer', 'def2-svp').nao == 48
