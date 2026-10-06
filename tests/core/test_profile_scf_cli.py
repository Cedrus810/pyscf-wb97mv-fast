import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.slow
def test_profile_cli_writes_json(tmp_path):
    out = tmp_path / 'p.json'
    subprocess.run([sys.executable, os.path.join(ROOT, 'benchmarks', 'profile_scf.py'),
                    'water_dimer', '--conv-tol', '1e-7', '--out', str(out)],
                   check=True, env={**os.environ, 'OMP_NUM_THREADS': '8'})
    r = json.loads(out.read_text())
    assert r['patched'] is True and r['converged'] is True
    assert r['timings']['k_lr'] > 0 and 'dm' not in r
