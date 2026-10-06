import numpy as np
import pytest
from pyscf import dft

from pyscf_wb97mv_fast.core import sgx_patch
from pyscf_wb97mv_fast.experimental.cosx.dynamic_ab import DynamicAB
from pyscf_wb97mv_fast.core.testsystems import build_mol


@pytest.mark.slow
def test_dynamic_ab_energy_unchanged():
    mol = build_mol('water_dimer')
    e = {}
    for use_dyn in (False, True):
        sgx_patch.apply()
        try:
            mf = dft.RKS(mol, xc='wb97m-v').COSX(pjs=True)
            mf.conv_tol = 1e-9
            if use_dyn:
                with DynamicAB(mf).attach():
                    e[use_dyn] = mf.kernel()
            else:
                e[use_dyn] = mf.kernel()
            assert mf.converged
        finally:
            sgx_patch.revert()
    assert abs(e[True] - e[False]) < 1e-8, e


@pytest.mark.slow
def test_dynamic_ab_switches_stage():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    mf.conv_tol = 1e-9
    sgx_patch.apply()
    try:
        dyn = DynamicAB(mf).attach()      # switch_at auto: 10*sqrt(conv_tol)
        mf.kernel()
        assert dyn.switch_cycle is not None
        # the initial-guess veff alone would give switch_cycle == 1 and one SR
        # build; real B-path SCF cycles must precede the switch
        assert dyn.switch_cycle >= 2, dyn.switch_cycle
        assert dyn._bpath.n_sr_builds >= 2, dyn._bpath.n_sr_builds
        assert mf.converged
    finally:
        sgx_patch.revert()


def test_switch_threshold_semantics():
    mf = dft.RKS(build_mol('water_dimer'), xc='wb97m-v').COSX(pjs=True)
    dyn = DynamicAB(mf, switch_at=1e-5).attach()
    try:
        assert dyn._bpath._attached           # starts on the B path
        dyn.rn = 1e-4
        assert dyn.rn > dyn.switch_at
        dyn.rn = 1e-6
        assert dyn.rn <= dyn.switch_at
    finally:
        dyn.remove()
