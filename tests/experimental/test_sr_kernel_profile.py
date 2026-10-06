from pyscf_wb97mv_fast.core.testsystems import build_mol
from sr_kernel_profile import profile_classes


def test_profile_classes_small():
    rows = profile_classes(build_mol('water_dimer'), 0.3, npts=500, repeat=1,
                           rbins=((0, 2), (5, 10)))
    kernels = {r['kernel'] for r in rows}
    assert kernels == {'full', 'LR', 'SR'}
    assert all(r['ns_per_int'] > 0 for r in rows)
    classes = {(r['li'], r['lj']) for r in rows}
    assert (0, 0) in classes and (1, 1) in classes and (2, 2) in classes
