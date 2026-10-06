"""Package layering (S1 Task 2): mainline never imports the frozen COSX code."""
import ast
import importlib
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, 'pyscf_wb97mv_fast')
MAINLINE = ('core', 'reference', 'staging', 'gpu')
FROZEN = ('tolerance', 'controller', 'sr_screening', 'dynamic_ab', 'fastgrad',
          'vv10_staged', 'acc', 'acc.jax_vv10', 'acc.hardware_profile')


def _imported_names(path):
    """Every module name a file can import: `import a.b`, `from a import b`
    (yields a.b, so `from pkg import experimental` is caught), relative forms,
    and string literals passed to importlib.import_module / __import__."""
    tree = ast.parse(open(path, encoding='utf-8').read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = ('.' * node.level) + (node.module or '')
            yield from (f'{mod}.{a.name}' if mod.strip('.') else mod + a.name
                        for a in node.names)
        elif isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
            if fname in ('import_module', '__import__') and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                yield node.args[0].value


def test_scanner_catches_all_import_forms(tmp_path):
    """The guard below is only as good as _imported_names."""
    src = tmp_path / 'm.py'
    src.write_text(
        'from pyscf_wb97mv_fast import experimental\n'
        'from .. import experimental as e\n'
        'import importlib\n'
        "importlib.import_module('pyscf_wb97mv_fast.experimental.cosx.tolerance')\n"
        "__import__('pyscf_wb97mv_fast.experimental')\n")
    names = list(_imported_names(str(src)))
    assert sum('experimental' in n for n in names) == 4, names


def test_mainline_does_not_import_experimental():
    offenders = []
    for sub in MAINLINE:
        for dirpath, _, files in os.walk(os.path.join(PKG, sub)):
            for f in files:
                if f.endswith('.py'):
                    p = os.path.join(dirpath, f)
                    offenders += [(p, n) for n in _imported_names(p) if 'experimental' in n]
    assert offenders == []


def test_package_import_is_cheap():
    """`import pyscf_wb97mv_fast` must not pull in pyscf (slow on NFS)."""
    code = 'import sys, pyscf_wb97mv_fast; print("pyscf" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.strip()
    assert out == 'False'


@pytest.mark.parametrize('name', FROZEN)
def test_frozen_module_lives_under_experimental(name):
    importlib.import_module(f'pyscf_wb97mv_fast.experimental.cosx.{name}')
    top = name.split('.')[0]
    assert importlib.util.find_spec(f'pyscf_wb97mv_fast.{top}') is None


def test_package_no_longer_exports_fast_path():
    import pyscf_wb97mv_fast
    assert not hasattr(pyscf_wb97mv_fast, 'fast_path')
