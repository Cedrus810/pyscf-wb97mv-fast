"""S5 Task 1: FP32 Boys function (gpu.boys), CUDA path vs scipy FP64."""
import numpy as np
import pytest
from scipy.special import gammainc, gamma

from pyscf_wb97mv_fast.gpu import backends


def ref_boys(n, T):
    """F_n(T) = gamma(n+1/2) * P(n+1/2, T) / (2 T^(n+1/2)); F_n(0) = 1/(2n+1)."""
    T = np.asarray(T, dtype=np.float64)
    out = np.empty_like(T)
    small = T < 1e-12
    out[small] = 1.0 / (2 * n + 1)
    Ts = T[~small]
    out[~small] = gamma(n + 0.5) * gammainc(n + 0.5, Ts) / (2.0 * Ts ** (n + 0.5))
    return out


@pytest.mark.gpu
def test_boys_fp32_relative_error():
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.boys import boys_eval
    T = np.concatenate([[0.0, 1e-30, 1e-8], np.linspace(0, 50, 20001), np.geomspace(50, 1e6, 2000)]).astype(np.float32)
    F = cp.asnumpy(boys_eval(cp, cp.asarray(T), 6))
    for n in range(7):
        ref = ref_boys(n, T.astype(np.float64))
        rel = np.abs(F[n] - ref) / ref
        assert np.all(np.isfinite(F[n]))
        assert rel.max() < 2e-7, (n, rel.max(), T[rel.argmax()])


@pytest.mark.gpu
def test_boys_fp32_unbiased():
    """Mean signed relative error over T in [0, 50] must be << FP32 eps:
    a biased F_n biases every integral the same way (cf. expf, FINDINGS.md).
    The range covers the asymptotic branch beyond the T = 36 switch, whose
    own truncation is a one-signed bias (5.3e-8 for F_6 at T = 30)."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.boys import boys_eval
    T = np.random.default_rng(0).uniform(0, 50, 1_000_000).astype(np.float32)
    F = cp.asnumpy(boys_eval(cp, cp.asarray(T), 6))
    for n in range(7):
        ref = ref_boys(n, T.astype(np.float64))
        assert abs(np.mean((F[n] - ref) / ref)) < 1e-8, n


@pytest.mark.gpu
def test_boys_fp32_nmax_slices():
    """nmax < 6 must agree with the nmax=6 call on the shared entries."""
    cp = backends.cupy_module()
    if cp is None: pytest.skip('no CUDA device')
    from pyscf_wb97mv_fast.gpu.boys import boys_eval
    T = np.linspace(0, 40, 101).astype(np.float32)
    full = cp.asnumpy(boys_eval(cp, cp.asarray(T), 6))
    for nmax in (0, 2, 4, 6):
        part = cp.asnumpy(boys_eval(cp, cp.asarray(T), nmax))
        assert part.shape == (nmax + 1, T.size)
        assert np.array_equal(part, full[:nmax + 1])
    with pytest.raises(ValueError):
        boys_eval(cp, cp.asarray(T), 7)
