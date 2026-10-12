"""Tau2 GRPO training on Tinker.

Package named tau2_train (not tinker) to avoid shadowing the tinker SDK.
"""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.realpath(os.path.join(_PKG_DIR, "..", ".."))

_TAU2_BENCH_SRC = os.path.join(_PROJECT_ROOT, "benchmark_refs", "tau2-bench", "src")
if os.path.isdir(_TAU2_BENCH_SRC) and _TAU2_BENCH_SRC not in sys.path:
    sys.path.insert(0, _TAU2_BENCH_SRC)
