#!/usr/bin/env python3
"""Embed BFCL rollouts v3: bundled tool calls with actual tool outputs.

Preserves the turn/step hierarchy from the executor's conversation:
- Tool-call steps keep bundled actions + concatenated tool return values
- None/text steps are preserved as turn boundaries with the next user prompt
- Normalizes DS text-format tool calls to match Qwen structured format

Usage::

    python run_bfcl_embed_v3.py --source qwen
    python run_bfcl_embed_v3.py --source dsv32
    python run_bfcl_embed_v3.py --source dsv32 --dry-run
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

# text-embedding-3-large has an 8192 token limit.
MAX_TEXT_CHARS = 8_000

from generation.bfcl.cluster_dag import (
    _get_turn_dicts,
    _step_to_action_text,
    embed_texts,
)


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


def _get_tool_responses_concat(step_entries: list[dict]) -> str:
    """Concatenate all tool responses from an inference log step."""
    parts = []
    for entry in step_entries:
        if entry.get("role") == "tool":
            content = entry.get("content", "")
            parts.append(str(content) if content else "<empty>")
    return "\n".join(parts) if parts else "<empty>"


def _normalize_action_text(step) -> str:
    """Normalize action text: extract tool calls from DS text format,
    or serialize Qwen structured calls. Keeps the bundled format."""
    if isinstance(step, list):
        # Qwen structured tool calls — use existing serializer
        return _step_to_action_text(step)
    elif isinstance(step, str):
        text = step.strip()
        if not text or text == "None":
            return "<none>"
        # Check if it's a DS-style bracketed tool call list
        # e.g. [ls(), cd(folder="document"), ...]
        if re.match(r'^\[.*\]$', text, re.DOTALL):
            return text
        # Text with embedded tool calls — keep as-is (narration)
        return text
    return "<none>"


# ── Rollout parsing ─────────────────────────────────────────────────────────


def parse_task_rollouts_v3(task_dir: Path, task_id: str) -> dict:
    """Parse rollouts preserving turn/step structure with real tool outputs.

    - Tool-call steps: action = bundled calls, response = concatenated tool returns
    - None/text steps: action = <none> or text, response = next user prompt
    """
    rollout_dirs = sorted(task_dir.glob("rollout_*"))
    if not rollout_dirs:
        raise FileNotFoundError(f"No rollout_* dirs in {task_dir}")

    action_texts: list[str] = []
    response_texts: list[str] = []
    indices: list[tuple[int, int]] = []
    # "tool_response" | "turn_end" (plain-text ending) | "decode_fail"
    # (call syntax present but nothing executed — silent failure)
    step_kinds: list[str] = []
    rewards: dict[int, float] = {}
    call_re = re.compile(r"\[[^\[\]]*\w+\s*\(")

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

                            action = _normalize_action_text(step)

                            # Check if this step has tool responses
                            has_tool_responses = any(
                                e.get("role") == "tool" for e in step_entries
                            )

                            if has_tool_responses:
                                resp = _get_tool_responses_concat(step_entries)
                            else:
                                resp = next_user_msg

                            raw = step if isinstance(step, str) else ""
                            if has_tool_responses:
                                kind = "tool_response"
                            elif call_re.search(raw):
                                kind = "decode_fail"
                            else:
                                kind = "turn_end"

                            action_texts.append(_truncate(action))
                            response_texts.append(_truncate(resp))
                            indices.append((rid, global_step))
                            step_kinds.append(kind)
                            global_step += 1

    return {
        "action_texts": action_texts,
        "response_texts": response_texts,
        "indices": indices,
        "step_kinds": step_kinds,
        "rewards": rewards,
    }


# ── Embedding ───────────────────────────────────────────────────────────────


def embed_and_save(task_id: str, category: str, runs_root: Path,
                   emb_dir: Path, v1_emb_dir: Path | None = None):
    """Embed v3: reuse action NPZs from v1, only embed new response texts."""
    import shutil

    task_dir = runs_root / category / task_id
    act_npz = emb_dir / f"{task_id}_action.npz"
    res_npz = emb_dir / f"{task_id}_response.npz"

    if act_npz.exists() and res_npz.exists():
        print(f"  SKIP {task_id}: embeddings already exist")
        return True

    # Copy action NPZ from v1 (same step structure, same action text)
    v1_act = v1_emb_dir / f"{task_id}_action.npz" if v1_emb_dir else None
    emb_dir.mkdir(parents=True, exist_ok=True)

    if v1_act and v1_act.exists() and not act_npz.exists():
        shutil.copy2(v1_act, act_npz)
        print(f"    Copied action NPZ from v1")
    elif not act_npz.exists():
        print(f"    WARNING: no v1 action NPZ for {task_id}, embedding from scratch")
        data = parse_task_rollouts_v3(task_dir, task_id)
        act_mat = embed_texts(data["action_texts"])
        act_arrays = {}
        for i, (rid, step) in enumerate(data["indices"]):
            act_arrays[f"r{rid}_s{step}"] = act_mat[i]
        np.savez_compressed(str(act_npz), **act_arrays)

    # Parse for response texts and rewards
    data = parse_task_rollouts_v3(task_dir, task_id)
    n_steps = len(data["action_texts"])
    n_rollouts = len(data["rewards"])
    if n_steps < 3:
        print(f"  SKIP {task_id}: too few steps ({n_steps})")
        return False

    pass_rate = sum(1 for v in data["rewards"].values() if v >= 0.5) / max(len(data["rewards"]), 1)
    print(f"  {task_id}: {n_steps} steps, {n_rollouts} rollouts, SR={pass_rate:.0%}")

    print(f"    Embedding responses ({n_steps} texts)...")
    res_mat = embed_texts(data["response_texts"])

    res_arrays = {}
    for i, (rid, step) in enumerate(data["indices"]):
        res_arrays[f"r{rid}_s{step}"] = res_mat[i]

    np.savez_compressed(str(res_npz), **res_arrays)

    # Copy rewards from v1 or save fresh
    rewards_path = emb_dir / f"{task_id}_rewards.json"
    v1_rewards = v1_emb_dir / f"{task_id}_rewards.json" if v1_emb_dir else None
    if v1_rewards and v1_rewards.exists():
        shutil.copy2(v1_rewards, rewards_path)
    else:
        with open(rewards_path, "w") as f:
            json.dump({str(k): v for k, v in data["rewards"].items()}, f)

    print(f"    Saved: {res_npz.name}")
    return True


# ── Dry run ─────────────────────────────────────────────────────────────────


def dry_run(task_id: str, category: str, runs_root: Path, max_rollouts: int = 3):
    task_dir = runs_root / category / task_id
    data = parse_task_rollouts_v3(task_dir, task_id)

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
        if len(act) > 120:
            act = act[:120] + "..."
        if len(resp) > 120:
            resp = resp[:120] + "..."
        print(f"    Step {step}: ACTION: {act}")
        print(f"             RESPONSE: {resp}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--source", type=str, default="qwen",
                        choices=list(SOURCES.keys()))
    parser.add_argument("--runs-root", type=Path, default=None, help="raw BFCL run dir (default: by --source under $PRECOG_DATA)")
    parser.add_argument("--emb-dir", type=Path, default=None, help="output dir for per-task npz (default: by --source)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-rollouts", type=int, default=3)
    args = parser.parse_args()

    src = SOURCES[args.source]
    runs_root = args.runs_root or src["runs_root"]
    emb_dir = args.emb_dir or src["emb_dir"]
    v1_emb_dir = None   # always embed actions from scratch in the release

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
            if embed_and_save(tid, cat, runs_root=runs_root, emb_dir=emb_dir, v1_emb_dir=v1_emb_dir):
                successes += 1
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)

    print(f"\nDone: {successes}/{len(tasks)} tasks embedded")


if __name__ == "__main__":
    main()
