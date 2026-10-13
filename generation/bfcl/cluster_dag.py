"""Build action-response DAGs from BFCL multi-turn rollouts.

Parses BFCL result JSONL, embeds actions and responses, clusters with
HDBSCAN, and renders trajectory-flow and global-graph visualizations —
reusing the Tau2 clustering infrastructure in ``src.prm``.

Usage::

    python -m src.bfcl_wrap.cluster_dag \\
        --run-dir exp/bfcl/runs/multi_turn_5x5 \\
        --task multi_turn_base_0 \\
        --output-dir exp/bfcl/runs/multi_turn_5x5/dag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 1. Parse BFCL results
# ---------------------------------------------------------------------------


def _step_to_action_text(step) -> str:
    """Convert a single BFCL result step to embeddable action text."""
    if isinstance(step, list):
        parts = []
        for tc in step:
            for name, args_json in tc.items():
                try:
                    args = json.loads(args_json)
                    sorted_args = ", ".join(
                        f"{k}={v!r}" for k, v in sorted(args.items())
                    )
                    parts.append(f"{name}({sorted_args})")
                except (json.JSONDecodeError, AttributeError):
                    parts.append(f"{name}({args_json})")
        return "\n".join(parts) if parts else "<empty>"
    if isinstance(step, str):
        return step.strip()[:500] or "<empty>"
    return "<empty>"


def _extract_tool_responses(step_entries: list[dict]) -> str:
    """Collect tool-role content from an inference_log step list."""
    parts = []
    for entry in step_entries:
        if entry.get("role") == "tool":
            parts.append(str(entry.get("content", "")))
    return "\n".join(parts) if parts else "<empty>"


def _get_turn_dicts(inference_log: list) -> list[dict]:
    """Return only the turn-trace dicts from inference_log (skip state_info)."""
    return [e for e in inference_log if isinstance(e, dict) and "begin_of_turn_query" in e]


def parse_task_rollouts(
    run_dir: Path,
    task_id: str,
    model_name: str,
) -> dict:
    """Parse all rollouts for one task across a BFCL run directory.

    Returns::

        {
            "action_texts": [...],
            "response_texts": [...],
            "indices": [(rollout_id, global_step), ...],
            "rewards": {rollout_id: 0.0 | 1.0, ...},
            "reasoning": [...],   # per-index reasoning_content
            "step_meta": [...],   # per-index metadata dicts
        }
    """
    rollout_dirs = sorted(run_dir.glob("rollout_*"))
    if not rollout_dirs:
        # category/task/rollout_* layout
        rollout_dirs = sorted(run_dir.glob("*/*/rollout_*"))
    if not rollout_dirs:
        # category/rollout_* layout (legacy)
        rollout_dirs = sorted(run_dir.glob("*/rollout_*"))
    if not rollout_dirs:
        raise FileNotFoundError(f"No rollout_* dirs in {run_dir}")

    action_texts: list[str] = []
    response_texts: list[str] = []
    indices: list[tuple[int, int]] = []
    rewards: dict[int, float] = {}
    reasoning: list[str] = []
    step_meta: list[dict] = []

    for rollout_id, rdir in enumerate(rollout_dirs):
        result_files = list(rdir.rglob("*_result.json"))
        if not result_files:
            continue

        score_files = list(rdir.rglob("*_score.json"))
        failed_ids = set()
        for sf in score_files:
            with open(sf, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    if "id" in entry and not entry.get("valid", True):
                        failed_ids.add(entry["id"])

        for rf in result_files:
            with open(rf, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("id") != task_id:
                        continue

                    result_turns = entry.get("result", [])
                    inference_log = entry.get("inference_log", [])
                    reasoning_turns = entry.get("reasoning_content", [])

                    passed = task_id not in failed_ids
                    rewards[rollout_id] = 1.0 if passed else 0.0

                    turn_dicts = _get_turn_dicts(inference_log)

                    global_step = 0
                    for turn_idx, turn_steps in enumerate(result_turns):
                        turn_dict = turn_dicts[turn_idx] if turn_idx < len(turn_dicts) else {}
                        step_keys = sorted(
                            [k for k in turn_dict if k.startswith("step_")]
                        )
                        next_user_msg = "<end>"
                        if turn_idx + 1 < len(turn_dicts):
                            nq = turn_dicts[turn_idx + 1].get("begin_of_turn_query", [])
                            if nq:
                                next_user_msg = nq[0].get("content", "<end>")

                        for step_idx, step in enumerate(turn_steps):
                            action = _step_to_action_text(step)
                            action_texts.append(action)

                            sk = f"step_{step_idx}"
                            step_entries = turn_dict.get(sk, [])
                            is_tool_step = isinstance(step, list)

                            if is_tool_step:
                                resp = _extract_tool_responses(step_entries)
                            else:
                                resp = next_user_msg

                            response_texts.append(resp)
                            indices.append((rollout_id, global_step))

                            rc = ""
                            if turn_idx < len(reasoning_turns):
                                turn_rc = reasoning_turns[turn_idx]
                                if isinstance(turn_rc, list) and step_idx < len(turn_rc):
                                    rc = turn_rc[step_idx] or ""
                            reasoning.append(rc)

                            step_meta.append({
                                "rollout": rollout_id,
                                "turn": turn_idx,
                                "step_in_turn": step_idx,
                                "global_step": global_step,
                                "type": "tool" if is_tool_step else "text",
                                "action_preview": action[:120],
                            })
                            global_step += 1

    return {
        "action_texts": action_texts,
        "response_texts": response_texts,
        "indices": indices,
        "rewards": rewards,
        "reasoning": reasoning,
        "step_meta": step_meta,
    }


# ---------------------------------------------------------------------------
# 2. Embed
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-large",
    max_workers: int = 4,
    chunk_size: int = 128,
) -> np.ndarray:
    """Embed texts using the OpenAI API. Returns [N x dim] float32."""
    from openai import OpenAI

    client = OpenAI()
    dim = 3072
    embeddings = np.zeros((len(texts), dim), dtype=np.float32)

    chunks = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        idxs = list(range(i, i + len(chunk)))
        chunks.append((idxs, chunk))

    def _embed_chunk(idxs, txts):
        import time as _time

        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=model, input=txts)
                return [
                    (idxs[item.index], np.array(item.embedding, dtype=np.float32))
                    for item in resp.data
                ]
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    _time.sleep(2**attempt)
                    continue
                raise
        resp = client.embeddings.create(model=model, input=txts)
        return [
            (idxs[item.index], np.array(item.embedding, dtype=np.float32))
            for item in resp.data
        ]

    from tqdm import tqdm

    pbar = tqdm(total=len(texts), desc="Embedding", unit="text")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_embed_chunk, idxs, txts): ci
            for ci, (idxs, txts) in enumerate(chunks)
        }
        for fut in as_completed(futures):
            results = fut.result()
            for idx, vec in results:
                embeddings[idx] = vec
            pbar.update(len(results))
    pbar.close()

    return embeddings


# ---------------------------------------------------------------------------
# 3. Cluster
# ---------------------------------------------------------------------------


def cluster_steps(
    action_mat: np.ndarray,
    response_mat: np.ndarray,
    min_cluster_size: int = 3,
    noise_strategy: str = "outlier_bin",
    similarity_method: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute paired similarity and run HDBSCAN. Returns (labels, S_paired)."""
    from src.prm.paired_action_response import (
        hdbscan_cluster_affinity,
        paired_similarity,
    )

    S_paired, _, _ = paired_similarity(action_mat, response_mat, method=similarity_method)
    labels = hdbscan_cluster_affinity(
        S_paired,
        min_cluster_size=min_cluster_size,
        noise_strategy=noise_strategy,
    )
    return labels, S_paired


# ---------------------------------------------------------------------------
# 4. DAG + visualization
# ---------------------------------------------------------------------------


def build_and_render(
    labels: np.ndarray,
    indices: list[tuple[int, int]],
    rewards: dict[int, float],
    output_dir: Path,
    task_id: str,
    step_meta: list[dict] | None = None,
    action_texts: list[str] | None = None,
    response_texts: list[str] | None = None,
):
    """Build global thought graph, render visualizations, write cluster JSON."""
    from src.prm.thought_dag import (
        build_global_thought_graph,
        render_global_graph,
        render_trajectory_flow,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    G, rollout_paths = build_global_thought_graph(labels, indices, rewards)
    K = int(labels.max()) + 1
    noise_lbl = K - 1

    flow_path = output_dir / f"{task_id}_flow.png"
    render_trajectory_flow(
        labels,
        indices,
        rewards,
        output_path=flow_path,
        title=f"{task_id} — Trajectory Flow (K={K})",
        noise_label=noise_lbl,
    )
    print(f"  Flow diagram: {flow_path}")

    graph_path = output_dir / f"{task_id}_graph.png"
    render_global_graph(G, graph_path, title=f"{task_id} — Thought-State Graph")
    print(f"  Graph: {graph_path}")

    cluster_data = {
        "task_id": task_id,
        "n_rollouts": len(set(r for r, _ in indices)),
        "n_steps": len(indices),
        "n_clusters": int(labels.max() + 1) if len(labels) > 0 else 0,
        "rewards": {str(k): v for k, v in rewards.items()},
        "rollout_paths": {str(k): v for k, v in rollout_paths.items()},
        "steps": [],
    }
    for i, ((rid, step), label) in enumerate(zip(indices, labels)):
        entry = {
            "rollout": rid,
            "global_step": step,
            "cluster": int(label),
        }
        if step_meta and i < len(step_meta):
            entry.update(step_meta[i])
        if action_texts and i < len(action_texts):
            entry["action_text"] = action_texts[i][:300]
        if response_texts and i < len(response_texts):
            entry["response_text"] = response_texts[i][:300]
        cluster_data["steps"].append(entry)

    clusters_path = output_dir / f"{task_id}_clusters.json"
    with open(clusters_path, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=2)
        f.write("\n")
    print(f"  Clusters: {clusters_path}")

    return G, rollout_paths


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------


def process_task(
    run_dir: Path,
    task_id: str,
    model_name: str,
    output_dir: Path,
    min_cluster_size: int = 3,
):
    """Full pipeline for one task: parse -> embed -> cluster -> DAG -> render."""
    print(f"\n--- Processing {task_id} ---")

    data = parse_task_rollouts(run_dir, task_id, model_name)
    n = len(data["action_texts"])
    n_rollouts = len(data["rewards"])
    print(f"  Parsed: {n} steps across {n_rollouts} rollouts")
    print(f"  Rewards: {data['rewards']}")

    if n < 3:
        print(f"  Skipping — too few steps ({n}) for clustering")
        return None

    print("  Embedding actions...")
    action_mat = embed_texts(data["action_texts"])
    print("  Embedding responses...")
    response_mat = embed_texts(data["response_texts"])

    labels, S = cluster_steps(
        action_mat, response_mat, min_cluster_size=min_cluster_size
    )
    n_clusters = int(labels.max() + 1) if len(labels) > 0 else 0
    print(f"  Clusters: {n_clusters}")

    G, paths = build_and_render(
        labels,
        data["indices"],
        data["rewards"],
        output_dir,
        task_id,
        step_meta=data["step_meta"],
        action_texts=data["action_texts"],
        response_texts=data["response_texts"],
    )

    np.savez_compressed(
        output_dir / f"{task_id}_embeddings.npz",
        action=action_mat,
        response=response_mat,
        labels=labels,
        similarity=S,
    )

    return {"task_id": task_id, "n_steps": n, "n_rollouts": n_rollouts, "n_clusters": n_clusters}


def discover_tasks(run_dir: Path) -> tuple[dict[str, Path], str]:
    """Find all task IDs, their base directories, and the model name.

    Returns ``(task_dirs, model_name)`` where *task_dirs* maps each task ID
    to the directory whose immediate children are ``rollout_*`` folders.

    Supported layouts (checked in order):

    1. ``run_dir/<category>/<task_id>/rollout_*``  (new canonical)
    2. ``run_dir/<category>/rollout_*``             (legacy category-flat)
    3. ``run_dir/rollout_*``                        (flat)
    """
    task_dirs: dict[str, Path] = {}
    model_name = ""

    def _detect_model(rollout_dir: Path) -> str:
        result_dir = rollout_dir / "result"
        if result_dir.exists():
            for d in result_dir.iterdir():
                if d.is_dir():
                    return d.name
        return ""

    # --- layout 1: category/task/rollout_* ---
    # Task directory name *is* the task ID in this layout.
    cat_task_rollouts = sorted(run_dir.glob("*/*/rollout_*"))
    if cat_task_rollouts:
        seen_task_dirs: set[Path] = set()
        for rdir in cat_task_rollouts:
            task_dir = rdir.parent
            if task_dir in seen_task_dirs:
                continue
            seen_task_dirs.add(task_dir)
            tid = task_dir.name
            task_dirs[tid] = task_dir
            if not model_name:
                model_name = _detect_model(rdir)
        if task_dirs:
            return task_dirs, model_name

    # --- layout 2 & 3: category/rollout_* or rollout_* ---
    rollout_dirs = sorted(run_dir.glob("*/rollout_*")) or sorted(run_dir.glob("rollout_*"))
    for rdir in rollout_dirs:
        if not model_name:
            model_name = _detect_model(rdir)
        result_dir = rdir / "result"
        if not result_dir.exists():
            continue
        for model_dir in result_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for rf in model_dir.rglob("*_result.json"):
                with open(rf, encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line)
                        tid = entry.get("id")
                        if tid and tid not in task_dirs:
                            task_dirs[tid] = run_dir

    return task_dirs, model_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="BFCL run directory containing rollout_*/ subdirs",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help="Task ID (e.g. multi_turn_base_0) or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/dag)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model subdirectory name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="HDBSCAN min_cluster_size (default: 3)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    explicit_output = args.output_dir is not None

    task_dirs, detected_model = discover_tasks(run_dir)
    model_name = args.model_name or detected_model
    if not model_name:
        print("Could not detect model name. Pass --model-name.", file=sys.stderr)
        sys.exit(1)

    if args.task == "all":
        task_ids = sorted(task_dirs.keys())
    else:
        task_ids = [args.task]
        if args.task not in task_dirs:
            task_dirs[args.task] = run_dir

    print(f"Run dir: {run_dir}")
    print(f"Model: {model_name}")
    print(f"Tasks ({len(task_ids)}): {task_ids}")

    summaries = []
    for tid in task_ids:
        task_base = task_dirs.get(tid, run_dir)
        if explicit_output:
            out = args.output_dir.resolve()
        else:
            out = task_base / "dag"

        result = process_task(
            task_base, tid, model_name, out, min_cluster_size=args.min_cluster_size
        )
        if result:
            result["output_dir"] = str(out)
            summaries.append(result)

    summary_path = (args.output_dir or run_dir).resolve() / "dag_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
        f.write("\n")
    print(f"\nSummary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
