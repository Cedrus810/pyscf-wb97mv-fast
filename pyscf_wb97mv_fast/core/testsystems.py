"""Standard synthetic test systems used by tests and benchmarks."""
import numpy as np
from pyscf import gto

WATER_DIMER = '''
O  -1.551007  -0.114520   0.000000
H  -1.934259   0.762503   0.000000
H  -0.599677   0.040712   0.000000
O   1.350625   0.111469   0.000000
H   1.680398  -0.373741  -0.758561
H   1.680398  -0.373741   0.758561
'''


def alkane_chain(n):
    atoms = []
    for i in range(n):
        x, y = 1.27 * i, 0.42 * (-1) ** i
        atoms += [('C', (x, y, 0)),
                  ('H', (x, y + (-1) ** i, 0.9)),
                  ('H', (x, y + (-1) ** i, -0.9))]
    atoms += [('H', (-1.0, 0.8, 0)),
              ('H', (1.27 * (n - 1) + 1.0, 0.8 * (-1) ** (n - 1), 0))]
    return atoms


def water_cube(m, spacing=3.1, seed=7):
    """m^3 waters on a jittered cubic lattice, random orientations, no clashes."""
    rng = np.random.default_rng(seed)
    oh, ang = 0.9572, np.deg2rad(104.52)
    local = np.array([[0, 0, 0],
                      [oh, 0, 0],
                      [oh * np.cos(ang), oh * np.sin(ang), 0]])
    atoms, placed = [], []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                for _ in range(200):
                    q = rng.normal(size=4)
                    q /= np.linalg.norm(q)
                    a, b, c, d = q
                    rot = np.array([
                        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                        [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                        [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])
                    origin = spacing * np.array([ix, iy, iz]) + rng.normal(scale=0.1, size=3)
                    xyz = local @ rot.T + origin
                    if not placed or np.min(np.linalg.norm(
                            np.array(placed)[:, None] - xyz[None], axis=-1)) > 1.6:
                        break
                placed.extend(xyz)
                atoms += [('O', xyz[0]), ('H', xyz[1]), ('H', xyz[2])]
    return atoms


def build_mol(system, basis='def2-svp', max_memory=16000):
    if system == 'water_dimer':
        atom = WATER_DIMER
    elif system.startswith('chain'):
        atom = alkane_chain(int(system[5:]))
    elif system.startswith('water'):
        m = round(int(system[5:]) ** (1 / 3))
        atom = water_cube(m)
    else:
        atom = system  # xyz file path
    return gto.M(atom=atom, basis=basis, verbose=0, max_memory=max_memory)
