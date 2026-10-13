"""Tau2 TDP generation pipeline."""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.realpath(os.path.join(_PKG_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# tau2-bench (sierra-research/tau2-bench, commit c5b2d22): either `pip install -e`
# it so `import tau2` works, or point TAU2_BENCH_SRC at its `src/` directory.
_TAU2_BENCH_SRC = os.environ.get("TAU2_BENCH_SRC")
if _TAU2_BENCH_SRC and os.path.isdir(_TAU2_BENCH_SRC) and _TAU2_BENCH_SRC not in sys.path:
    sys.path.insert(0, _TAU2_BENCH_SRC)
