"""S1 Task 3: frozen benchmarks moved to benchmarks/experimental/cosx, one results dir."""
import glob
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COSX = os.path.join(ROOT, 'benchmarks', 'experimental', 'cosx')


def _load(path):
    name = 'bench_' + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_results_dir_is_repo_benchmarks_results():
    paths = _load(os.path.join(ROOT, 'benchmarks', '_paths.py'))
    assert paths.REPO_ROOT == ROOT
    assert paths.results_dir() == os.path.join(ROOT, 'benchmarks', 'results')
    assert os.path.isdir(paths.results_dir())


def test_add_repo_to_syspath_is_idempotent():
    paths = _load(os.path.join(ROOT, 'benchmarks', '_paths.py'))
    paths.add_repo_to_syspath()
    paths.add_repo_to_syspath()
    assert sys.path.count(ROOT) >= 1
    assert sys.path.count(ROOT) == len([p for p in sys.path if p == ROOT])


def test_old_location_is_gone():
    assert not os.path.exists(os.path.join(ROOT, 'benchmarks', 'sgx_locality'))


@pytest.mark.parametrize('script', sorted(glob.glob(os.path.join(COSX, '*.py')))
                         or ['<no frozen benchmark scripts found>'])
def test_frozen_benchmark_imports(script, tmp_path):
    """Import each script in a fresh isolated interpreter (-I: no cwd, no
    PYTHONPATH) from outside the repo, so only the script's own sys.path
    setup can make its imports resolve (conftest paths do not leak in)."""
    assert os.path.isfile(script), script
    code = ('import importlib.util as u; s = u.spec_from_file_location("m", %r); '
            'm = u.module_from_spec(s); s.loader.exec_module(m); print("ok")' % script)
    r = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().endswith('ok'), r.stderr[-2000:]


def test_shim_initializes_in_new_location():
    assert os.path.isfile(os.path.join(COSX, 'libcint_shim.so'))
    code = ('import sys; sys.path.insert(0, %r); import sgx_locality as s; '
            'from pyscf_wb97mv_fast.core.testsystems import build_mol; '
            's.Shim(build_mol("water_dimer")); print("ok")' % COSX)
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    assert out.strip().endswith('ok')


def test_analyze_reproduces_findings_numbers(capsys):
    analyze = _load(os.path.join(COSX, 'analyze.py'))
    analyze.main([os.path.join(COSX, 'result_chain30_def2-svp.json')])
    out = capsys.readouterr().out
    # FINDINGS.md section 4.1, chain30 stock row
    assert 'ints 0.568' in out
    assert 'per-int cost SR/LR 1.67' in out
