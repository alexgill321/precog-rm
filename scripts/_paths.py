"""Single source of truth for where data lives.

Everything large (embeddings, caches, checkpoints, GRPO logs) is read from the
data root; everything small that the paper's aggregation scripts read or write
(per-task result JSONs, sweep curves, figures) lives inside the repo so the
"no data needed" fast path keeps working.

    PRECOG_DATA   data root (default: <repo>/data). Populate it with
                  `python scripts/download.py --stage ...` or point it at a
                  directory with the layout documented in data/README.md.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = Path(os.environ.get("PRECOG_DATA", REPO / "data")).expanduser().resolve()

# ---- large inputs (HuggingFace data release) ----
TABLE1 = DATA / "table1_data"          # <benchmark>_<model>/ embedding corpora
CACHES = DATA / "caches"               # slim/, mmap/ (built by build_slim_caches + --convert)
SLIM_DIR = CACHES / "slim"
MMAP_DIR = CACHES / "mmap"
CKPTS = DATA / "ckpts"                 # precog/, ablations/, prefix/
CKPT_PRECOG = CKPTS / "precog"
CKPT_ABL = CKPTS / "ablations"
CKPT_PREFIX = CKPTS / "prefix"
RUNS = DATA / "runs"                   # 21 official GRPO runs
REPLAY = DATA / "replay"               # simulated early-stopping results_*.json
GRPO_REPLAY = DATA / "grpo_replay"     # baseline-run rollouts + embeddings for replay
EVAL = DATA / "eval"                   # base-model test-split eval
REASONING_EMB = DATA / "reasoning_emb"
ROLLOUTS = DATA / "rollouts"           # raw/, resample/, meta/

# ---- small, shipped in the repo ----
ABL = REPO / "exp" / "results" / "protonet_ablations"   # result JSONs + precog_parts
SPLITS = ABL / "task_splits.json"
FIGS = REPO / "paper" / "figures"
TABLES = REPO / "paper" / "tables"
GRPO_CACHE = REPO / "notebooks" / "data" / "tau2_es"     # rollout_cache/*.parquet, calibration.json
AC_META = REPO / "benchmark_refs" / "AgentClinic"        # tool-reliance lists

DATASETS = ["agentclinic", "bfclv3", "tau2"]
MODELS = ["gpt5mini", "qwen", "deepseek"]
BENCH_NAME = {"tau2": "Tau2-Bench", "bfclv3": "BFCLv3", "agentclinic": "AgentClinic"}
MODEL_NAME = {"deepseek": "DS-V3.2", "qwen": "Qwen3-80B", "gpt5mini": "GPT-5-mini"}


def require(path: Path, what: str, stage: str) -> Path:
    if not path.exists():
        raise SystemExit(f"missing {what}: {path}\n"
                         f"  -> run `python scripts/download.py --stage {stage}` "
                         f"or set PRECOG_DATA (currently {DATA})")
    return path
