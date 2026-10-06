"""gpu.warmup: one small staged SCF compiles every kernel."""
import pytest

from pyscf_wb97mv_fast.gpu import backends, warmup


def test_fallback_messages_are_recognised():
    msgs = ['pyscf_wb97mv_fast GPU get_k_only failed (Unsupported: l > 2); '
            'falling back to the CPU implementation\n...',
            'CuPy/CUDA not available; GPU fast path not installed',
            'some unrelated DeprecationWarning']
    assert warmup.fallbacks(msgs) == msgs[:2]


@pytest.mark.gpu
@pytest.mark.slow
def test_warmup_runs_all_gpu_paths():
    cupy = backends.cupy_module()
    if cupy is None or cupy.cuda.runtime.getDeviceCount() < 1:
        pytest.skip('no CUDA device')
    rc, report = warmup.warmup()
    assert rc == 0, report
    assert report[-1] == 'kernel cache warm'
