import numpy as np
from pyscf import dft
from pyscf.scf import diis as scf_diis

from pyscf_wb97mv_fast.core.hooks import HookSet, reset_diis, track_residual
from pyscf_wb97mv_fast.core.testsystems import build_mol


def _sym(rng, n=4):
    a = rng.normal(size=(n, n))
    return a + a.T


def _fed(diis, rng, n_updates):
    """Feed n_updates random (S, D, F) triples; return the last extrapolation."""
    s = np.eye(4)
    out = None
    for _ in range(n_updates):
        out = diis.update(s, _sym(rng), _sym(rng))
    return out


def test_reset_clears_history_and_behaves_like_fresh():
    used = scf_diis.CDIIS()
    _fed(used, np.random.default_rng(0), 3)
    assert len(used._bookkeep) == 3
    reset_diis(used)
    assert used._bookkeep == [] and used._head == 0 and used._H is None
    assert used._xprev is None and used._buffer == {}
    # next update must match a brand-new CDIIS fed the same data
    x_used = _fed(used, np.random.default_rng(1), 1)
    x_fresh = _fed(scf_diis.CDIIS(), np.random.default_rng(1), 1)
    assert np.allclose(x_used, x_fresh, atol=1e-14, rtol=0)


def test_reset_keeps_configuration():
    corth = np.eye(4)
    d = scf_diis.CDIIS(Corth=corth)
    d.space = 4
    _fed(d, np.random.default_rng(2), 2)
    reset_diis(d)
    assert d.space == 4 and d.Corth is corth


def test_reset_none_is_noop():
    reset_diis(None)


def test_reset_mid_scf_converges_to_same_energy():
    mol = build_mol('water_dimer', 'def2-svp')
    ref = dft.RKS(mol, xc='pbe')
    ref.conv_tol = 1e-10
    e_ref = ref.kernel()

    mf = dft.RKS(mol, xc='pbe')
    mf.conv_tol = 1e-10
    calls = []

    def factory(prev):
        def callback(envs):
            if envs['cycle'] == 2:          # 3rd cycle (0-based)
                reset_diis(envs['mf_diis'])
                calls.append(envs['cycle'])
            if callable(prev):
                prev(envs)
        return callback

    with HookSet() as hooks:
        hooks.wrap(mf, 'callback', factory)
        e = mf.kernel()
    assert calls == [2]
    assert mf.converged
    assert abs(e - e_ref) < 1e-8


def test_track_residual_still_exported():
    assert callable(track_residual)
