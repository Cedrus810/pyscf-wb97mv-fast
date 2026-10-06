"""Final-review fixes: packaging config and job scripts stay consistent with the layout."""
import os
import re
import tomllib

from setuptools import find_packages

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _package_dirs():
    base = os.path.join(ROOT, 'pyscf_wb97mv_fast')
    out = []
    for dirpath, dirnames, files in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        if '__init__.py' in files:
            out.append(os.path.relpath(dirpath, ROOT).replace(os.sep, '.'))
    return sorted(out)


def _configured_packages():
    cfg = tomllib.load(open(os.path.join(ROOT, 'pyproject.toml'), 'rb'))['tool']['setuptools']
    if 'packages' in cfg and isinstance(cfg['packages'], list):
        return sorted(cfg['packages'])
    find = cfg['packages']['find']
    return sorted(find_packages(where=ROOT, include=find.get('include', ['*']),
                                exclude=find.get('exclude', [])))


def test_every_subpackage_is_shipped():
    """`pip install .` must ship core/, reference/, ... not only the top package."""
    assert _configured_packages() == _package_dirs()


def test_verify_tests_job_runs_experimental_tests():
    """The pre-check before gates_water27.pbs (frozen benches) must not skip them."""
    text = open(os.path.join(ROOT, 'jobs', 'verify_tests.pbs')).read()
    calls = [l for l in text.splitlines() if re.search(r'-m pytest\b', l)]
    assert calls, 'no pytest call found'
    assert all('--experimental' in l for l in calls), calls
