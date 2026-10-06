import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, 'tests')
for p in (ROOT, TESTS, os.path.join(ROOT, 'benchmarks'),
          os.path.join(ROOT, 'benchmarks', 'experimental', 'cosx')):
    if p not in sys.path:
        sys.path.insert(0, p)

from pytest_markers import *  # noqa: E402,F401,F403  (experimental / gpu / slow hooks)
