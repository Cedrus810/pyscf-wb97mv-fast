import numpy as np
import pytest

from pyscf_wb97mv_fast.experimental.cosx.acc import hardware_profile
from pyscf_wb97mv_fast.experimental.cosx.acc import jax_vv10

try:
    import jax  # noqa: F401
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

B, C = 6.0, 0.01    # wb97m-v VV10 parameters


def _random_grid(n=200, seed=3):
    rng = np.random.default_rng(seed)
    coords = rng.normal(scale=4.0, size=(n, 3))
    rho = rng.uniform(0.5, 2.0, size=n)          # all above threshold
    grad2 = rng.uniform(0.0, 5.0, size=n)
    w = rng.uniform(0.01, 0.1, size=n)
    return coords, rho, grad2, w


def test_energy_matches_pyscf_vv10nlc():
    """The real validation: same grid, same parameters, pyscf's own routine."""
    coords, rho, grad2, w = _random_grid()
    M, mask = jax_vv10.build_kernel(coords, rho, grad2, b=B, C=C)
    assert mask.all()
    e = jax_vv10.energy(M, rho * w, b=B)
    e_ref = jax_vv10.pyscf_energy(coords, rho, grad2, w, b=B, C=C)
    assert abs(e - e_ref) < 1e-10 * abs(e_ref), (e, e_ref)


def test_energy_matches_pyscf_with_thresholded_points():
    coords, rho, grad2, w = _random_grid()
    rho[::3] = 1e-12                              # below threshold
    M, mask = jax_vv10.build_kernel(coords, rho, grad2, b=B, C=C)
    e = jax_vv10.energy(M, (rho * w)[mask], b=B)
    e_ref = jax_vv10.pyscf_energy(coords, rho, grad2, w, b=B, C=C)
    assert abs(e - e_ref) < 1e-10 * abs(e_ref), (e, e_ref)


def test_kernel_matches_direct_loop():
    coords, rho, grad2, w = _random_grid(n=60)
    M, _ = jax_vv10.build_kernel(coords, rho, grad2, b=B, C=C)
    W0, K = jax_vv10.local_quantities(rho, grad2, b=B, C=C)
    for i in range(0, 60, 7):
        for j in range(0, 60, 5):
            R2 = ((coords[i] - coords[j]) ** 2).sum()
            g = R2 * W0[i] + K[i]
            gp = R2 * W0[j] + K[j]
            assert M[i, j] == pytest.approx(1 / (g * gp * (g + gp)), rel=1e-10)


def test_mask_thresholds_points():
    coords, rho, grad2, w = _random_grid()
    rho[::3] = 1e-12
    M, mask = jax_vv10.build_kernel(coords, rho, grad2, b=B, C=C)
    assert M.shape == (int(mask.sum()), int(mask.sum()))
    assert not mask[::3].any()


def test_low_rank_energy():
    coords, rho, grad2, w = _random_grid(n=150)
    M, _ = jax_vv10.build_kernel(coords, rho, grad2, b=B, C=C)
    q = rho * w
    e_full = jax_vv10.energy(M, q, b=B)
    U, lam = jax_vv10.low_rank(M, tol=0.0)          # full rank: exact
    assert abs(jax_vv10.energy_low_rank(U, lam, q, b=B) - e_full) < 1e-12 * abs(e_full)
    U, lam = jax_vv10.low_rank(M, tol=1e-10)
    assert U.shape[1] <= len(rho)
    assert abs(jax_vv10.energy_low_rank(U, lam, q, b=B) - e_full) < 1e-6 * abs(e_full)


@pytest.mark.skipif(not HAS_JAX, reason='jax not installed')
def test_jax_backend_matches_numpy():
    coords, rho, grad2, w = _random_grid(n=120)
    Mn, _ = jax_vv10.build_kernel(coords, rho, grad2, b=B, C=C)
    Mj, _ = jax_vv10.build_kernel(coords, rho, grad2, b=B, C=C, backend='jax')
    assert np.abs(Mn - np.asarray(Mj)).max() < 1e-12


def test_hardware_profile_runs():
    res = hardware_profile.profile(repeat=1)
    assert res['jax'] is None or HAS_JAX
    for name in ('gemm_f64', 'gemm_f32', 'skinny_f64'):
        assert res['cpu'][name]['gflops'] > 0
