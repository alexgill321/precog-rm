"""Run BFCL ``generate`` then ``evaluate`` for multiple rollouts.

Supports two execution modes:
  - **In-process** (default): Builds one BFCL handler and runs inference
    via ThreadPoolExecutor.  Much faster for Vertex MaaS models because
    the OpenAI client / GCP auth token / HTTP connections are shared.
  - **Subprocess** (``--subprocess``): Original behaviour — spawns a
    separate ``bfcl generate`` + ``bfcl evaluate`` subprocess per rollout.

Usage:
    python -m src.bfcl_wrap.run \\
        --ids-file generation/bfcl/presets/mixed_200_tasks.json \\
        --model deepseek-v3.2-vertex \\
        --rollouts 50 --max-concurrency 30

With multiple rollouts, each repetition writes results to
``<output_dir>/<category>/<task_id>/rollout_NN/``.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore", message=".*end user credentials.*")
warnings.filterwarnings("ignore", message=".*quota project.*")


class _TqdmOnlyStream(io.TextIOBase):
    """Write-only stream that discards everything except tqdm bar updates.

    BFCL's handler prints per-step progress (``ID: ..., Turn: ..., Step: ...``)
    directly to stdout via ``print()``.  This wrapper swallows those lines so
    only our own tqdm bar (written to the real stderr) is visible.

    An optional log file captures all discarded output for debugging.
    """

    def __init__(self, log_path: Path | None = None):
        self._log = open(log_path, "a", encoding="utf-8") if log_path else None

    def write(self, s: str) -> int:
        if self._log:
            self._log.write(s)
            self._log.flush()
        return len(s)

    def flush(self) -> None:
        if self._log:
            self._log.flush()

    def close(self) -> None:
        if self._log:
            self._log.close()

# Models that need the patched bfcl entry point (Vertex MaaS + OpenRouter).
_VERTEX_MODELS = {
    "qwen3-next-80b-thinking-vertex",
    "qwen3-next-80b-vertex",
    "deepseek-v3.2-vertex",
    "gpt-5-nano-openrouter",
    "gpt-5-mini-openrouter",
}

# $/M tokens — same table as TRAJECT-Bench CostTracker
_PRICING: dict[str, dict[str, float]] = {
    "qwen3-next-80b-thinking-vertex": {"input": 0.15, "output": 1.20},
    "qwen3-next-80b-vertex": {"input": 0.15, "output": 1.20},
    "deepseek-v3.2-vertex": {"input": 0.30, "output": 1.20},
    "gpt-5-nano-openrouter": {"input": 0.05, "output": 0.40},
    "gpt-5-mini-openrouter": {"input": 0.25, "output": 2.00},
    "gpt-5-mini-2025-08-07-FC": {"input": 0.25, "output": 2.00},
    "gpt-5-mini-2025-08-07": {"input": 0.25, "output": 2.00},
    "gpt-5-nano-2025-08-07-FC": {"input": 0.05, "output": 0.40},
    "gpt-5-nano-2025-08-07": {"input": 0.05, "output": 0.40},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
}


def _safe_sum(val) -> int | float:
    """Recursively sum nested lists/ints, treating None as 0."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, list):
        return sum(_safe_sum(v) for v in val)
    return 0


def _tally_result_dir(result_dir: Path) -> dict:
    """Walk all ``*_result.json`` files under *result_dir* and sum tokens."""
    input_tokens = 0
    output_tokens = 0
    for rfile in result_dir.rglob("*_result.json"):
        with open(rfile, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                input_tokens += _safe_sum(entry.get("input_token_count", 0))
                output_tokens += _safe_sum(entry.get("output_token_count", 0))
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = _PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens / 1e6) * p["input"] + (output_tokens / 1e6) * p["output"]


def _repo_flow_eval_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_bfcl_binary() -> str:
    if os.environ.get("BFCL_BIN"):
        return os.environ["BFCL_BIN"]
    venv_bin = Path(sys.executable).parent / "bfcl"
    if venv_bin.is_file():
        return str(venv_bin)
    return "bfcl"


def _bfcl_command_prefix(model: str, bfcl_bin: str) -> list[str]:
    if model in _VERTEX_MODELS:
        return [sys.executable, "-m", "src.bfcl_wrap.bfcl_patched"]
    return [bfcl_bin]


def load_ids_file(path: Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("test_case_ids file must be a JSON object mapping category -> list of ids")
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f"Category {k!r} must map to a list of string ids")
        out[str(k)] = v
    return out


def categories_from_ids(ids: dict[str, list[str]]) -> list[str]:
    return sorted(ids.keys())


def write_test_case_ids(project_root: Path, ids: dict[str, list[str]]) -> None:
    project_root.mkdir(parents=True, exist_ok=True)
    dest = project_root / "test_case_ids_to_generate.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(ids, f, indent=2)
        f.write("\n")


def ensure_dotenv(project_root: Path, dotenv_source: Path | None) -> None:
    dest = project_root / ".env"
    if dest.exists():
        return
    if dotenv_source is not None and dotenv_source.is_file():
        shutil.copy2(dotenv_source, dest)
        return
    dest.write_text(
        "# API keys for the model providers you use with bfcl-eval.\n"
        "# OPENAI_API_KEY=sk-...\n",
        encoding="utf-8",
    )


def _load_dotenv_into_dict(env: dict[str, str], dotenv_path: Path) -> None:
    if not dotenv_path.is_file():
        return
    with open(dotenv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            # Skip empty values so a placeholder line (e.g. ``OPENROUTER_API_KEY=``)
            # never clobbers a real value already present in the inherited env.
            if key and val:
                env[key] = val


def _rollout_is_complete(project_root: Path) -> bool:
    has_result = any(project_root.glob("result/**/*_result.json"))
    has_score = any(project_root.glob("score/**/*_score.json"))
    return has_result and has_score


# ── Subprocess mode (original) ──────────────────────────────────────────────


def run_rollout_subprocess(
    *,
    bfcl_bin: str,
    project_root: Path,
    model: str,
    ids: dict[str, list[str]],
    temperature: float,
    num_threads: int | None,
    include_input_log: bool,
    exclude_state_log: bool,
    allow_overwrite: bool,
    skip_generate: bool,
    skip_evaluate: bool,
    dry_run: bool,
) -> None:
    write_test_case_ids(project_root, ids)
    cats = categories_from_ids(ids)
    test_category_arg = ",".join(cats)

    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(project_root.resolve())
    _load_dotenv_into_dict(env, project_root / ".env")

    prefix = _bfcl_command_prefix(model, bfcl_bin)

    gen = [
        *prefix, "generate",
        "--model", model,
        "--run-ids",
        "--temperature", str(temperature),
    ]
    if num_threads is not None:
        gen += ["--num-threads", str(num_threads)]
    if include_input_log:
        gen.append("--include-input-log")
    if exclude_state_log:
        gen.append("--exclude-state-log")
    if allow_overwrite:
        gen.append("--allow-overwrite")
    gen += ["--test-category", test_category_arg]

    ev = [
        *prefix, "evaluate",
        "--model", model,
        "--test-category", test_category_arg,
        "--partial-eval",
    ]

    if dry_run:
        print("[dry-run] BFCL_PROJECT_ROOT=", env["BFCL_PROJECT_ROOT"])
        print("[dry-run] ", " ".join(gen))
        print("[dry-run] ", " ".join(ev))
        return

    log_file = project_root / "bfcl.log"
    with open(log_file, "a", encoding="utf-8") as log:
        if not skip_generate:
            subprocess.run(gen, env=env, check=True, stdout=log, stderr=log)
        if not skip_evaluate:
            subprocess.run(ev, env=env, check=True, stdout=log, stderr=log)


def _run_work_item_subprocess(
    *,
    category: str,
    task_id: str,
    rollout_z: int,
    output_dir: Path,
    bfcl_bin: str,
    model: str,
    temperature: float,
    num_threads: int | None,
    include_input_log: bool,
    exclude_state_log: bool,
    allow_overwrite: bool,
    skip_generate: bool,
    skip_evaluate: bool,
    dry_run: bool,
    dotenv_source: Path | None,
) -> dict:
    project_root = output_dir / category / task_id / f"rollout_{rollout_z:02d}"
    project_root.mkdir(parents=True, exist_ok=True)

    if not allow_overwrite and _rollout_is_complete(project_root):
        return {
            "category": category, "task_id": task_id, "rollout": rollout_z,
            "input_tokens": 0, "output_tokens": 0, "skipped": True,
        }

    if not (project_root / ".env").exists() and (output_dir / ".env").exists():
        shutil.copy2(output_dir / ".env", project_root / ".env")
    ensure_dotenv(project_root, dotenv_source)

    run_rollout_subprocess(
        bfcl_bin=bfcl_bin,
        project_root=project_root,
        model=model,
        ids={category: [task_id]},
        temperature=temperature,
        num_threads=num_threads,
        include_input_log=include_input_log,
        exclude_state_log=exclude_state_log,
        allow_overwrite=allow_overwrite,
        skip_generate=skip_generate,
        skip_evaluate=skip_evaluate,
        dry_run=dry_run,
    )

    tally = {"input_tokens": 0, "output_tokens": 0}
    if not dry_run and not skip_generate:
        tally = _tally_result_dir(project_root / "result")

    return {
        "category": category, "task_id": task_id, "rollout": rollout_z,
        "input_tokens": tally["input_tokens"],
        "output_tokens": tally["output_tokens"],
        "skipped": False,
    }


# ── In-process mode ─────────────────────────────────────────────────────────


def _load_all_test_entries(ids: dict[str, list[str]]) -> dict[str, dict]:
    """Load BFCL test entries and index by test case ID.

    Returns a dict mapping test_case_id -> test_entry dict.
    """
    from bfcl_eval.utils import load_dataset_entry

    index: dict[str, dict] = {}
    for category, task_ids in ids.items():
        entries = load_dataset_entry(category)
        wanted = set(task_ids)
        for entry in entries:
            if entry["id"] in wanted:
                index[entry["id"]] = entry
    return index


def _run_work_item_inprocess(
    *,
    handler,
    test_entry: dict,
    category: str,
    task_id: str,
    rollout_z: int,
    output_dir: Path,
    model: str,
    include_input_log: bool,
    exclude_state_log: bool,
    allow_overwrite: bool,
    bfcl_bin: str,
    dotenv_source: Path | None,
) -> dict:
    """Run one (task, rollout) using the shared handler in-process."""
    project_root = output_dir / category / task_id / f"rollout_{rollout_z:02d}"
    project_root.mkdir(parents=True, exist_ok=True)

    if not allow_overwrite and _rollout_is_complete(project_root):
        return {
            "category": category, "task_id": task_id, "rollout": rollout_z,
            "input_tokens": 0, "output_tokens": 0, "skipped": True,
        }

    # Deep-copy test entry since inference mutates it
    test_case = copy.deepcopy(test_entry)

    # Give this rollout a unique id so that execute_multi_turn_func_call
    # creates isolated environment instances in globals() instead of
    # sharing state across concurrent rollouts of the same task.
    original_id = test_case["id"]
    test_case["id"] = f"{original_id}__r{rollout_z:02d}"

    from bfcl_eval._llm_response_generation import multi_threaded_inference
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )

    try:
        result_dict = multi_threaded_inference(
            handler, test_case, include_input_log, exclude_state_log,
        )
    finally:
        # Clean up globals entries to avoid memory leaks.
        rollout_id = test_case["id"]
        model_key = handler.model_name_underline_replaced
        for cls in test_entry.get("involved_classes", []):
            key = f"{model_key}_{rollout_id}_{cls}_instance"
            key = key.replace("-", "_").replace(".", "_").replace("/", "_").replace(":", "_")
            globals().pop(key, None)

    # Restore original id for correct result writing and evaluation
    result_dict["id"] = original_id

    # Write result to the per-rollout directory
    result_dir = project_root / "result"
    handler.write(result_dict, result_dir=result_dir, update_mode=True)

    # Tally tokens from the result
    input_tokens = _safe_sum(result_dict.get("input_token_count", 0))
    output_tokens = _safe_sum(result_dict.get("output_token_count", 0))

    # Run evaluation as subprocess (fast, no API calls)
    _run_evaluate_subprocess(
        project_root=project_root,
        model=model,
        category=category,
        bfcl_bin=bfcl_bin,
        dotenv_source=dotenv_source,
    )

    return {
        "category": category, "task_id": task_id, "rollout": rollout_z,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "skipped": False,
    }


def _run_evaluate_subprocess(
    *,
    project_root: Path,
    model: str,
    category: str,
    bfcl_bin: str,
    dotenv_source: Path | None,
) -> None:
    """Run bfcl evaluate as a subprocess (fast, no API calls)."""
    if not (project_root / ".env").exists():
        ensure_dotenv(project_root, dotenv_source)

    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = str(project_root.resolve())
    _load_dotenv_into_dict(env, project_root / ".env")

    prefix = _bfcl_command_prefix(model, bfcl_bin)
    ev = [
        *prefix, "evaluate",
        "--model", model,
        "--test-category", category,
        "--partial-eval",
    ]

    log_file = project_root / "bfcl.log"
    with open(log_file, "a", encoding="utf-8") as log:
        try:
            subprocess.run(ev, env=env, check=True, stdout=log, stderr=log,
                           timeout=120)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log.write(f"\nEvaluate failed: {e}\n")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Run directory (default: $PRECOG_DATA/rollouts/raw/bfclv3/<run_name>)",
    )
    parser.add_argument(
        "--run-name", type=str, default="local",
        help="Run name under $PRECOG_DATA/rollouts/raw/bfclv3/ when --output-dir is omitted",
    )
    parser.add_argument(
        "--ids-file", type=Path, required=True,
        help="JSON mapping test category -> list of test ids",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument(
        "--max-concurrency", type=int, default=1,
        help="Max concurrent work items (threads in-process, subprocesses otherwise)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--include-input-log", action="store_true")
    parser.add_argument("--exclude-state-log", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dotenv", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--bfcl-bin", default=_default_bfcl_binary(),
        help="bfcl executable (default: bfcl or BFCL_BIN)",
    )
    parser.add_argument(
        "--use-subprocess", action="store_true",
        help="Use subprocess per rollout (original slow mode) instead of in-process",
    )
    args = parser.parse_args()

    root = _repo_flow_eval_root()
    os.chdir(root)

    ids = load_ids_file(args.ids_file.resolve())
    categories = categories_from_ids(ids)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (Path(os.environ.get("PRECOG_DATA", root / "data")) / "rollouts" / "raw" / "bfclv3" / args.run_name).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    dotenv_src = args.dotenv.resolve() if args.dotenv else None
    ensure_dotenv(output_dir, dotenv_src)

    manifest = {
        "model": args.model,
        "ids_file": str(args.ids_file),
        "ids": ids,
        "rollouts": args.rollouts,
        "max_concurrency": args.max_concurrency,
        "temperature": args.temperature,
        "num_threads": args.num_threads,
        "include_input_log": args.include_input_log,
        "mode": "subprocess" if args.use_subprocess else "inprocess",
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    if args.rollouts < 1:
        print("--rollouts must be >= 1", file=sys.stderr)
        sys.exit(1)

    # Build flat work queue
    work_items: list[tuple[str, str, int]] = []
    total_tasks = 0
    for cat in categories:
        for tid in ids[cat]:
            total_tasks += 1
            for z in range(args.rollouts):
                work_items.append((cat, tid, z))

    n_items = len(work_items)
    max_workers = min(args.max_concurrency, n_items) or 1
    use_inprocess = not args.use_subprocess

    # ── In-process setup ──────────────────────────────────────────────────
    handler = None
    test_entry_index = None

    if use_inprocess and not args.dry_run:
        # Set a temporary BFCL_PROJECT_ROOT before importing bfcl modules.
        # The actual per-rollout paths are handled by handler.write calls.
        tmp_root = output_dir / "_bfcl_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        os.environ["BFCL_PROJECT_ROOT"] = str(tmp_root)
        ensure_dotenv(tmp_root, dotenv_src)

        # Load dotenv credentials into the current process so in-process handlers
        # that read env vars directly (e.g. OpenRouter → OPENROUTER_API_KEY) can
        # see them. Subprocess mode merges the dotenv into the child env instead.
        _dotenv_vars: dict[str, str] = {}
        _load_dotenv_into_dict(_dotenv_vars, tmp_root / ".env")
        os.environ.update({k: v for k, v in _dotenv_vars.items() if v})

        # Patch model config for Vertex models
        if args.model in _VERTEX_MODELS:
            from src.bfcl_wrap.bfcl_patched import _patch_model_config
            _patch_model_config()

        from bfcl_eval._llm_response_generation import build_handler

        handler = build_handler(args.model, args.temperature)
        print(f"Built handler: {type(handler).__name__} (in-process mode)", flush=True)

        test_entry_index = _load_all_test_entries(ids)
        print(f"Loaded {len(test_entry_index)} test entries", flush=True)

    print(
        f"Work items: {n_items} ({total_tasks} tasks x {args.rollouts} rollouts), "
        f"max_concurrency={max_workers}, mode={'inprocess' if use_inprocess else 'subprocess'}",
        flush=True,
    )

    # Thread-safe accumulators
    lock = threading.Lock()
    cumulative_input = 0
    cumulative_output = 0
    completed = 0
    skipped = 0
    failed: list[tuple[str, str, int, str]] = []
    t0 = time.time()

    # Redirect stdout to swallow BFCL's per-step print() output.
    # tqdm writes to stderr by default, so the progress bar stays visible.
    _real_stdout = sys.stdout
    if use_inprocess and not args.dry_run:
        sys.stdout = _TqdmOnlyStream(log_path=output_dir / "bfcl_verbose.log")

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_items, desc="BFCL", unit="item", file=_real_stdout)
    except ImportError:
        pbar = None

    def _on_complete(result: dict) -> None:
        nonlocal cumulative_input, cumulative_output, completed, skipped
        with lock:
            cumulative_input += result["input_tokens"]
            cumulative_output += result["output_tokens"]
            completed += 1
            if result.get("skipped"):
                skipped += 1
            cost = _estimate_cost(args.model, cumulative_input, cumulative_output)
            elapsed = time.time() - t0
            tag = " (cached)" if result.get("skipped") else ""
            msg = (
                f"[{completed}/{n_items}] {result['category']}/{result['task_id']} "
                f"r{result['rollout']:02d}{tag}: "
                f"{result['input_tokens']:,} in / {result['output_tokens']:,} out | "
                f"cumulative ${cost:.4f} ({elapsed:.0f}s)"
            )
            if pbar:
                pbar.set_postfix_str(f"${cost:.4f} ({skipped} cached)")
                pbar.update(1)
            else:
                _real_stdout.write(f"  {msg}\n")
                _real_stdout.flush()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for cat, tid, z in work_items:
            if use_inprocess and not args.dry_run:
                entry = test_entry_index.get(tid)
                if entry is None:
                    print(f"  WARNING: no test entry for {cat}/{tid}, skipping",
                          flush=True)
                    continue
                fut = executor.submit(
                    _run_work_item_inprocess,
                    handler=handler,
                    test_entry=entry,
                    category=cat,
                    task_id=tid,
                    rollout_z=z,
                    output_dir=output_dir,
                    model=args.model,
                    include_input_log=args.include_input_log,
                    exclude_state_log=args.exclude_state_log,
                    allow_overwrite=args.allow_overwrite,
                    bfcl_bin=args.bfcl_bin,
                    dotenv_source=dotenv_src,
                )
            else:
                fut = executor.submit(
                    _run_work_item_subprocess,
                    category=cat,
                    task_id=tid,
                    rollout_z=z,
                    output_dir=output_dir,
                    bfcl_bin=args.bfcl_bin,
                    model=args.model,
                    temperature=args.temperature,
                    num_threads=args.num_threads,
                    include_input_log=args.include_input_log,
                    exclude_state_log=args.exclude_state_log,
                    allow_overwrite=args.allow_overwrite,
                    skip_generate=args.skip_generate,
                    skip_evaluate=args.skip_evaluate,
                    dry_run=args.dry_run,
                    dotenv_source=dotenv_src,
                )
            futures[fut] = (cat, tid, z)

        for fut in as_completed(futures):
            cat, tid, z = futures[fut]
            try:
                result = fut.result()
                _on_complete(result)
            except Exception as exc:
                with lock:
                    completed += 1
                    failed.append((cat, tid, z, str(exc)))
                if pbar:
                    pbar.update(1)
                _real_stdout.write(f"  FAILED {cat}/{tid} r{z:02d}: {exc}\n")
                _real_stdout.flush()

    if pbar:
        pbar.close()

    # Restore stdout
    if sys.stdout is not _real_stdout:
        sys.stdout.close()
        sys.stdout = _real_stdout

    total_cost = _estimate_cost(args.model, cumulative_input, cumulative_output)
    elapsed_total = time.time() - t0

    if not args.dry_run:
        cost_summary = {
            "model": args.model,
            "categories": categories,
            "rollouts": args.rollouts,
            "max_concurrency": args.max_concurrency,
            "mode": "subprocess" if args.use_subprocess else "inprocess",
            "input_tokens": cumulative_input,
            "output_tokens": cumulative_output,
            "total_tokens": cumulative_input + cumulative_output,
            "estimated_cost_usd": round(total_cost, 6),
            "pricing_per_M_tokens": _PRICING.get(args.model, {}),
            "elapsed_seconds": round(elapsed_total, 1),
            "failed": [{"category": c, "task_id": t, "rollout": z, "error": e}
                       for c, t, z, e in failed],
        }
        with open(output_dir / "cost_summary.json", "w", encoding="utf-8") as f:
            json.dump(cost_summary, f, indent=2)
            f.write("\n")

        print(f"\n{'='*60}")
        print(f"Done: {total_tasks} tasks x {args.rollouts} rollout(s), model={args.model}")
        if skipped:
            print(f"  Skipped (already complete): {skipped}/{n_items}")
        print(
            f"  Tokens: {cumulative_input:,} input + {cumulative_output:,} output "
            f"= {cumulative_input + cumulative_output:,} total"
        )
        print(f"  Estimated cost: ${total_cost:.4f}")
        print(f"  Wall clock: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
        if failed:
            print(f"  Failed: {len(failed)} work items")
            for c, t, z, e in failed[:10]:
                print(f"    {c}/{t} rollout {z}: {e}")
            if len(failed) > 10:
                print(f"    ... and {len(failed) - 10} more")
        print(f"  Cost summary: {output_dir / 'cost_summary.json'}")
        print(f"{'='*60}")
    else:
        print("Done (dry run).")


if __name__ == "__main__":
    main()
