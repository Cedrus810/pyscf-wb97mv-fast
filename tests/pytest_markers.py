"""Pytest hooks for this repo's markers; star-imported by tests/conftest.py.

- experimental: every test under an `experimental/` directory (relative to the
  rootdir) gets it automatically, and all experimental tests are skipped unless
  `--experimental` is given -- whatever `-m` selects.
- gpu: skipped when CuPy cannot be imported or no CUDA device is visible.
- slow: long-running (full SCF, water27-size systems, numpy VV10, finite
  differences, anything over about 10 s).  Skipped unless `--slow` is given,
  so the default run stays short.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption('--experimental', action='store_true', default=False,
                     help='also run tests under experimental/ (frozen COSX-side code)')
    parser.addoption('--slow', action='store_true', default=False,
                     help='also run tests marked slow (full SCF, large systems; minutes each)')


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: runs a full SCF or takes more than about one minute')
    config.addinivalue_line('markers', 'experimental: frozen COSX-side test; opt-in via --experimental')
    config.addinivalue_line('markers', 'gpu: needs CuPy and a CUDA GPU; skipped otherwise')


def _gpu_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # ImportError, CUDARuntimeError, driver problems
        return False


def _under_experimental_dir(item, rootpath):
    try:
        parts = item.path.relative_to(rootpath).parts[:-1]
    except ValueError:
        parts = item.path.parts[:-1]
    return 'experimental' in parts


@pytest.hookimpl(tryfirst=True)   # mark before `-m` selection runs
def pytest_collection_modifyitems(config, items):
    run_experimental = config.getoption('--experimental')
    run_slow = config.getoption('--slow')
    skip_experimental = pytest.mark.skip(reason='experimental (frozen); use --experimental')
    skip_slow = pytest.mark.skip(reason='slow; use --slow')
    skip_gpu = pytest.mark.skip(reason='no CUDA GPU / CuPy')
    gpu_ok = None
    for item in items:
        if _under_experimental_dir(item, config.rootpath):
            item.add_marker(pytest.mark.experimental)
        if item.get_closest_marker('experimental') and not run_experimental:
            item.add_marker(skip_experimental)
        if item.get_closest_marker('slow') and not run_slow:
            item.add_marker(skip_slow)
        if item.get_closest_marker('gpu'):
            if gpu_ok is None:
                gpu_ok = _gpu_available()
            if not gpu_ok:
                item.add_marker(skip_gpu)
