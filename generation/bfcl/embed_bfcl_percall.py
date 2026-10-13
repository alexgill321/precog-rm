#!/usr/bin/env python3
"""Embed BFCL rollouts with per-tool-call splitting.

Each individual tool call becomes its own (action, response) step in the
embedding sequence, matching the granularity of tau2 rollouts. Tool calls
are normalized to a common format regardless of whether the model used
structured calls (Qwen) or text-embedded calls (DeepSeek).

Usage::

    python run_bfcl_embed_v2.py --source qwen              # Qwen rollouts
    python run_bfcl_embed_v2.py --source dsv32              # DeepSeek rollouts
    python run_bfcl_embed_v2.py --source dsv32 --dry-run    # preview parsing without embedding
    python run_bfcl_embed_v2.py --source qwen --task multi_turn_base_0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

import sys
REPO = Path(__file__).resolve().parents[2]            # precog-rm root
sys.path.insert(0, str(REPO))
PRESETS = REPO / "generation" / "bfcl" / "presets"
PRESET_PATH = PRESETS / "mixed_200_tasks.json"

DATA = Path(os.environ.get("PRECOG_DATA", REPO / "data"))
# source name -> (default raw-run root, default embedding output dir) under $PRECOG_DATA
SOURCES = {
    "qwen":     {"runs_root": DATA / "rollouts/raw/bfclv3/qwen",     "emb_dir": DATA / "bfcl_embeddings/qwen"},
    "dsv32":    {"runs_root": DATA / "rollouts/raw/bfclv3/deepseek", "emb_dir": DATA / "bfcl_embeddings/deepseek"},
    "gpt5mini": {"runs_root": DATA / "rollouts/raw/bfclv3/gpt5mini", "emb_dir": DATA / "bfcl_embeddings/gpt5mini"},
}

from generation.bfcl.cluster_dag import (
    _get_turn_dicts,
    embed_texts,
)

# text-embedding-3-large has an 8192 token limit.
# Repeated punctuation like '""""""' tokenizes at ~1 char/token,
# so we need a conservative char limit.
MAX_TEXT_CHARS = 8_000


def _truncate(text: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        return text[:MAX_TEXT_CHARS]
    return text


def _rollout_id_from_dir(rdir: Path) -> int:
    m = re.search(r"rollout_(\d+)", rdir.name)
    if not m:
        raise ValueError(f"Cannot parse rollout ID from {rdir.name}")
    return int(m.group(1))


def _read_reward(score_files: list[Path], task_id: str) -> float:
    for sf in score_files:
        with open(sf, encoding="utf-8") as f:
            first_line = json.loads(f.readline())
            if first_line.get("total_count", 0) > 0:
                return 1.0 if first_line["correct_count"] == first_line["total_count"] else 0.0
    return 0.0


# ── Tool call extraction ────────────────────────────────────────────────────


def _normalize_call(name: str, args_str: str) -> str:
    """Normalize a tool call to 'name(key=value, ...)' format."""
    try:
        args = json.loads(args_str)
        if isinstance(args, dict):
            parts = ", ".join(f'{k}={v!r}' for k, v in sorted(args.items()))
            return f"{name}({parts})"
    except (json.JSONDecodeError, TypeError):
        pass
    return f"{name}({args_str})"


def _extract_calls_from_structured(step: list[dict]) -> list[str]:
    """Extract individual calls from a Qwen-style structured tool call list.

    Each element is like {"cd": '{"folder": "document"}'}.
    """
    calls = []
    for tc in step:
        if isinstance(tc, dict):
            for name, args_json in tc.items():
                calls.append(_normalize_call(name, args_json))
    return calls


def _extract_calls_from_text(text: str) -> list[str]:
    """Extract individual calls from DS-style text containing [func(...)] patterns."""
    pattern = r'(\w+)\(([^)]*)\)'
    matches = re.findall(pattern, text)
    calls = []
    for name, args_raw in matches:
        # Build a JSON-like args dict from keyword arguments
        try:
            # Try parsing as Python-style kwargs: key="value", key2="value2"
            arg_pairs = re.findall(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\S+)',
                                   args_raw)
            if arg_pairs:
                args_dict = {}
                for k, v in arg_pairs:
                    v = v.strip("'\"")
                    args_dict[k] = v
                parts = ", ".join(f'{k}={v!r}' for k, v in sorted(args_dict.items()))
                calls.append(f"{name}({parts})")
            elif not args_raw.strip():
                calls.append(f"{name}()")
            else:
                calls.append(f"{name}({args_raw})")
        except Exception:
            calls.append(f"{name}({args_raw})")
    return calls


def _get_tool_responses(step_entries: list[dict]) -> list[str]:
    """Extract individual tool responses from inference log step entries.

    Returns one response string per tool call, matched by position.
    """
    responses = []
    for entry in step_entries:
        if entry.get("role") == "tool":
            content = entry.get("content", "")
            responses.append(str(content) if content else "<empty>")
    return responses


# ── Rollout parsing ─────────────────────────────────────────────────────────


def parse_task_rollouts_v2(task_dir: Path, task_id: str) -> dict:
    """Parse rollouts with per-tool-call splitting.

    Each individual tool call becomes its own step in the sequence.
    Text-only steps (agent narration/summary) are kept as single steps.
    """
    rollout_dirs = sorted(task_dir.glob("rollout_*"))
    if not rollout_dirs:
        raise FileNotFoundError(f"No rollout_* dirs in {task_dir}")

    action_texts: list[str] = []
    response_texts: list[str] = []
    indices: list[tuple[int, int]] = []
    rewards: dict[int, float] = {}

    for rdir in rollout_dirs:
        rid = _rollout_id_from_dir(rdir)

        result_files = list(rdir.rglob("*_result.json"))
        if not result_files:
            continue

        score_files = list(rdir.rglob("*_score.json"))
        rewards[rid] = _read_reward(score_files, task_id)

        for rf in result_files:
            with open(rf, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("id") != task_id:
                        continue

                    result_turns = entry.get("result", [])
                    inference_log = entry.get("inference_log", [])
                    turn_dicts = _get_turn_dicts(inference_log)

                    global_step = 0
                    for turn_idx, turn_steps in enumerate(result_turns):
                        turn_dict = turn_dicts[turn_idx] if turn_idx < len(turn_dicts) else {}
                        next_user_msg = "<end>"
                        if turn_idx + 1 < len(turn_dicts):
                            nq = turn_dicts[turn_idx + 1].get("begin_of_turn_query", [])
                            if nq:
                                next_user_msg = nq[0].get("content", "<end>")

                        for step_idx, step in enumerate(turn_steps):
                            sk = f"step_{step_idx}"
                            step_entries = turn_dict.get(sk, [])

                            if isinstance(step, list):
                                # Structured tool calls (Qwen)
                                calls = _extract_calls_from_structured(step)
                                tool_resps = _get_tool_responses(step_entries)
                            elif isinstance(step, str) and step and step.strip() \
                                    and step.strip() != "None":
                                # Text step — check for embedded tool calls (DS)
                                calls = _extract_calls_from_text(step)
                                tool_resps = _get_tool_responses(step_entries)
                                if not calls:
                                    # Pure narration/summary — keep as one step
                                    action_texts.append(_truncate(step.strip()))
                                    response_texts.append(_truncate(next_user_msg))
                                    indices.append((rid, global_step))
                                    global_step += 1
                                    continue
                            elif step is None:
                                continue
                            else:
                                continue

                            # Split into individual (call, response) pairs
                            for call_idx, call_text in enumerate(calls):
                                action_texts.append(_truncate(call_text))
                                if call_idx < len(tool_resps):
                                    response_texts.append(_truncate(tool_resps[call_idx]))
                                else:
                                    response_texts.append("<empty>")
                                indices.append((rid, global_step))
                                global_step += 1

    return {
        "action_texts": action_texts,
        "response_texts": response_texts,
        "indices": indices,
        "rewards": rewards,
    }


# ── Embedding ───────────────────────────────────────────────────────────────


def embed_and_save(task_id: str, category: str, runs_root: Path,
                   emb_dir: Path):
    task_dir = runs_root / category / task_id
    act_npz = emb_dir / f"{task_id}_action.npz"
    res_npz = emb_dir / f"{task_id}_response.npz"

    if act_npz.exists() and res_npz.exists():
        print(f"  SKIP {task_id}: embeddings already exist")
        return True

    data = parse_task_rollouts_v2(task_dir, task_id)
    n_steps = len(data["action_texts"])
    n_rollouts = len(data["rewards"])
    if n_steps < 3:
        print(f"  SKIP {task_id}: too few steps ({n_steps})")
        return False

    pass_rate = sum(1 for v in data["rewards"].values() if v >= 0.5) / max(len(data["rewards"]), 1)
    print(f"  {task_id}: {n_steps} steps, {n_rollouts} rollouts, SR={pass_rate:.0%}")

    print(f"    Embedding actions ({n_steps} texts)...")
    act_mat = embed_texts(data["action_texts"])
    print(f"    Embedding responses ({n_steps} texts)...")
    res_mat = embed_texts(data["response_texts"])

    act_arrays = {}
    res_arrays = {}
    for i, (rid, step) in enumerate(data["indices"]):
        key = f"r{rid}_s{step}"
        act_arrays[key] = act_mat[i]
        res_arrays[key] = res_mat[i]

    emb_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(act_npz), **act_arrays)
    np.savez_compressed(str(res_npz), **res_arrays)

    rewards_path = emb_dir / f"{task_id}_rewards.json"
    with open(rewards_path, "w") as f:
        json.dump({str(k): v for k, v in data["rewards"].items()}, f)

    print(f"    Saved: {act_npz.name}, {res_npz.name}")
    return True


# ── Dry run ─────────────────────────────────────────────────────────────────


def dry_run(task_id: str, category: str, runs_root: Path, max_rollouts: int = 3):
    """Print parsed steps without embedding — for verification."""
    task_dir = runs_root / category / task_id
    data = parse_task_rollouts_v2(task_dir, task_id)

    rids_seen = set()
    for rid, _ in data["indices"]:
        rids_seen.add(rid)

    print(f"  {task_id}: {len(data['action_texts'])} steps across "
          f"{len(data['rewards'])} rollouts")

    shown = 0
    current_rid = None
    for i, (rid, step) in enumerate(data["indices"]):
        if rid != current_rid:
            if shown >= max_rollouts:
                break
            current_rid = rid
            shown += 1
            reward = data["rewards"].get(rid, "?")
            print(f"\n  --- Rollout {rid} (reward={reward}) ---")

        act = data["action_texts"][i]
        resp = data["response_texts"][i]
        if len(resp) > 120:
            resp = resp[:120] + "..."
        print(f"    Step {step}: ACTION: {act}")
        print(f"             RESPONSE: {resp}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, default=None, help="Single task ID to embed")
    parser.add_argument("--category", type=str, default=None, help="Single category to embed")
    parser.add_argument("--source", type=str, default="qwen",
                        choices=list(SOURCES.keys()),
                        help="Which rollout source to embed (default: qwen)")
    parser.add_argument("--runs-root", type=Path, default=None, help="raw BFCL run dir (default: by --source under $PRECOG_DATA)")
    parser.add_argument("--emb-dir", type=Path, default=None, help="output dir for per-task npz (default: by --source)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print parsed steps without embedding")
    parser.add_argument("--dry-run-rollouts", type=int, default=3,
                        help="Number of rollouts to show in dry run")
    args = parser.parse_args()

    src = SOURCES[args.source]
    runs_root = args.runs_root or src["runs_root"]
    emb_dir = args.emb_dir or src["emb_dir"]

    if args.source == "dsv32":
        preset = {}
        for cat_dir in sorted(runs_root.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
                continue
            if cat_dir.name not in ("multi_turn_base", "multi_turn_long_context",
                                     "multi_turn_miss_func", "multi_turn_miss_param"):
                continue
            task_ids = sorted(d.name for d in cat_dir.iterdir() if d.is_dir())
            if task_ids:
                preset[cat_dir.name] = task_ids
    else:
        preset = json.loads(PRESET_PATH.read_text())

    tasks: list[tuple[str, str]] = []
    for cat, task_ids in preset.items():
        if args.category and cat != args.category:
            continue
        for tid in task_ids:
            if args.task and tid != args.task:
                continue
            tasks.append((tid, cat))

    if args.dry_run:
        print(f"Dry run: {len(tasks)} tasks ({args.source})")
        for i, (tid, cat) in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {cat}/{tid}")
            try:
                dry_run(tid, cat, runs_root, max_rollouts=args.dry_run_rollouts)
            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
        return

    print(f"Embedding {len(tasks)} tasks ({args.source}) → {emb_dir}")
    successes = 0
    for i, (tid, cat) in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {cat}/{tid}")
        try:
            if embed_and_save(tid, cat, runs_root=runs_root, emb_dir=emb_dir):
                successes += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print(f"\nDone: {successes}/{len(tasks)} tasks embedded")


if __name__ == "__main__":
    main()
