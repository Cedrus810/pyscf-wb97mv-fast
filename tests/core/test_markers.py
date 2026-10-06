"""S1 Task 4: `experimental` tests are opt-in (--experimental), so are `slow` ones (--slow);
`gpu` tests skip without a GPU.

Each case builds a throw-away test tree whose conftest loads the same hook
module the repo uses (tests/pytest_markers.py) and runs pytest in a subprocess.
"""
import os
import subprocess
import sys
import textwrap

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(tmp_path, files, *args, env_extra=None):
    (tmp_path / 'conftest.py').write_text(
        f'import sys\nsys.path.insert(0, {TESTS_DIR!r})\nfrom pytest_markers import *  # noqa\n')
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body))
    env = dict(os.environ, **(env_extra or {}))
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider', *args],
                       cwd=tmp_path, capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


PASSING = 'def test_ok():\n    assert True\n'
SLOW_PASSING = 'import pytest\n\n@pytest.mark.slow\ndef test_ok():\n    assert True\n'


def test_plain_tests_still_run(tmp_path):
    code, out = _run(tmp_path, {'test_plain.py': PASSING})
    assert code == 0 and '1 passed' in out, out


def test_experimental_skipped_by_default(tmp_path):
    code, out = _run(tmp_path, {'experimental/test_x.py': PASSING})
    assert '1 skipped' in out and 'use --experimental' in out, out


def test_experimental_not_pulled_in_by_marker_selection(tmp_path):
    code, out = _run(tmp_path, {'experimental/test_x.py': SLOW_PASSING}, '-m', 'slow', '--slow')
    assert '1 skipped' in out and '1 passed' not in out, out


def test_experimental_runs_with_option(tmp_path):
    code, out = _run(tmp_path, {'experimental/test_x.py': PASSING}, '--experimental')
    assert code == 0 and '1 passed' in out, out


def test_slow_skipped_by_default(tmp_path):
    code, out = _run(tmp_path, {'test_s.py': SLOW_PASSING})
    assert '1 skipped' in out and 'use --slow' in out and '1 passed' not in out, out


def test_slow_runs_with_option(tmp_path):
    code, out = _run(tmp_path, {'test_s.py': SLOW_PASSING}, '--slow')
    assert code == 0 and '1 passed' in out, out


def test_gpu_skipped_without_cupy(tmp_path):
    fake = tmp_path / 'fakemods' / 'cupy'
    fake.mkdir(parents=True)
    (fake / '__init__.py').write_text("raise ImportError('fake: no cupy')\n")
    body = 'import pytest\n\n@pytest.mark.gpu\ndef test_needs_gpu():\n    assert True\n'
    code, out = _run(tmp_path, {'test_gpu.py': body},
                     env_extra={'PYTHONPATH': str(tmp_path / 'fakemods')})
    assert '1 skipped' in out and 'no CUDA GPU' in out, out
