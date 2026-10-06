"""Repository paths shared by all benchmark scripts, wherever they live.

Every benchmark writes its JSON/log output to results_dir()
(<repo>/benchmarks/results), not next to the script.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_dir():
    path = os.path.join(REPO_ROOT, 'benchmarks', 'results')
    os.makedirs(path, exist_ok=True)
    return path


def add_repo_to_syspath():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
