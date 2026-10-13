#!/usr/bin/env python3
"""Embed a tau2 rollout cell (action and observation text per step) with
text-embedding-3-large into <data-dir>/embeddings/{action,response}/<task>.npz,
the layout every downstream loader reads. Builds the task list from the
rollout JSON files on disk. Needs OPENAI_API_KEY.

Usage:
  python -m generation.tau2.embed_tau2 --data-dir $PRECOG_DATA/table1_data/tau2_deepseek            # dry run
  python -m generation.tau2.embed_tau2 --data-dir $PRECOG_DATA/table1_data/tau2_deepseek --apply    # embed
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
from src.prm.embeddings import (  # noqa: E402
    extract_response_text,
    extract_sample_text,
    task_id_to_filename,
)

DATA_DIR = None          # set from --data-dir (a tau2 cell: {airline,retail,telecom}/<idx>/rollout_NNN.json)
OUT_EMB = None

MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
BATCH_SIZE = 2048


def _discover_tasks() -> list[dict]:
    """Scan the data directory and build a task list from rollout files."""
    tasks = []
    for domain in ["airline", "retail", "telecom"]:
        domain_dir = DATA_DIR / domain
        if not domain_dir.exists():
            continue
        for task_dir in sorted(domain_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_num = task_dir.name
            rollout_files = sorted(task_dir.glob("rollout_*.json"))
            if not rollout_files:
                continue
            rids = []
            for rf in rollout_files:
                rid = int(rf.stem.replace("rollout_", ""))
                rids.append(rid)
            label = f"{domain}/{task_num}"
            tasks.append({
                "label": label,
                "domain": domain,
                "task_dir": task_dir,
                "rids": rids,
                "n_rollouts": len(rids),
            })
    return tasks


def _embed_batch(client, texts: list[str]) -> list[np.ndarray]:
    resp = client.embeddings.create(input=texts, model=MODEL)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def _embed_task(client, task: dict, text_mode: str) -> dict[str, np.ndarray]:
    """Read all rollout files for a task and embed them."""
    unique_texts: dict[str, int] = {}
    text_list: list[str] = []
    task_structure: list[tuple[int, list[list[int]]]] = []

    for rid in task["rids"]:
        rollout_file = task["task_dir"] / f"rollout_{rid:03d}.json"
        if not rollout_file.exists():
            continue

        with open(rollout_file) as fh:
            raw = json.load(fh)
        all_steps = raw.get("steps", [])
        rollout_steps: list[list[int]] = []

        for step_i, step in enumerate(all_steps):
            step_indices: list[int] = []
            if text_mode == "response":
                text = extract_response_text(
                    all_steps, step_i, messages=raw.get("messages")
                )
                if text not in unique_texts:
                    unique_texts[text] = len(text_list)
                    text_list.append(text)
                step_indices.append(unique_texts[text])
            else:
                for sample in step.get("samples", []):
                    text = extract_sample_text(sample, text_mode)
                    if text not in unique_texts:
                        unique_texts[text] = len(text_list)
                        text_list.append(text)
                    step_indices.append(unique_texts[text])
            rollout_steps.append(step_indices)

        task_structure.append((rid, rollout_steps))
        del raw

    embeddable = [(i, t) for i, t in enumerate(text_list) if t != "<empty>"]
    embeddings: dict[int, np.ndarray] = {}
    zero_vec = np.zeros(EMBED_DIM, dtype=np.float32)
    empty_idx = unique_texts.get("<empty>")

    for batch_start in range(0, len(embeddable), BATCH_SIZE):
        batch = embeddable[batch_start : batch_start + BATCH_SIZE]
        batch_texts = [t for _, t in batch]
        vecs = _embed_batch(client, batch_texts)
        for (idx, _), vec in zip(batch, vecs):
            embeddings[idx] = vec

    arrays: dict[str, np.ndarray] = {}
    for rid, steps in task_structure:
        for s_idx, sample_indices in enumerate(steps):
            key = f"r{rid}_s{s_idx}"
            vecs = []
            for text_idx in sample_indices:
                if text_idx == empty_idx:
                    vecs.append(zero_vec)
                else:
                    vecs.append(embeddings.get(text_idx, zero_vec))
            if vecs:
                arrays[key] = np.stack(vecs)
            else:
                arrays[key] = np.zeros((0, EMBED_DIM), dtype=np.float32)

    return arrays


def main():
    global DATA_DIR, OUT_EMB
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--data-dir", default=None,
                        help="Consolidated rollout dir (default: dsv32_200/data). "
                             "Embeddings are written to <data-dir>/embeddings.")
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR = Path(args.data_dir).resolve()
        OUT_EMB = DATA_DIR / "embeddings"

    tasks = _discover_tasks()
    if not tasks:
        print("No tasks found. Check that rollout files exist in "
              f"{DATA_DIR}/{{airline,retail,telecom}}/*/")
        return

    total_rollouts = sum(t["n_rollouts"] for t in tasks)
    print(f"Discovered {len(tasks)} tasks, {total_rollouts} rollouts")

    to_embed = []
    already_done = []

    for task in tasks:
        stem = task_id_to_filename(task["label"])
        out_act = OUT_EMB / "action" / f"{stem}.npz"
        out_res = OUT_EMB / "response" / f"{stem}.npz"
        if out_act.exists() and out_res.exists():
            npz = np.load(out_res)
            embedded_rids = {int(k.split("_")[0][1:]) for k in npz.keys()}
            npz.close()
            if len(embedded_rids) >= task["n_rollouts"]:
                already_done.append(task["label"])
            else:
                to_embed.append(task)
        else:
            to_embed.append(task)

    print(f"Already embedded: {len(already_done)}")
    print(f"Need embedding:   {len(to_embed)}")

    if to_embed:
        need_rollouts = sum(t["n_rollouts"] for t in to_embed)
        print(f"\nTasks to embed ({need_rollouts} rollouts):")
        for t in to_embed:
            print(f"  {t['label']}: {t['n_rollouts']} rollouts")

    if not args.apply:
        print("\nDRY RUN — use --apply to execute.")
        return

    from openai import OpenAI

    os.makedirs(OUT_EMB / "action", exist_ok=True)
    os.makedirs(OUT_EMB / "response", exist_ok=True)

    client = OpenAI()
    total_keys = 0

    for i, task in enumerate(to_embed):
        t0 = time.time()
        stem = task_id_to_filename(task["label"])

        for text_mode, subdir in [("action", "action"), ("response", "response")]:
            arrays = _embed_task(client, task, text_mode)
            out_path = OUT_EMB / subdir / f"{stem}.npz"
            np.savez_compressed(str(out_path), **arrays)
            total_keys += len(arrays)

        dt = time.time() - t0
        print(
            f"  [{i+1}/{len(to_embed)}] {task['label']:<40}  "
            f"{task['n_rollouts']} rollouts  {dt:.1f}s"
        )

    print(f"\nDone. Embedded {len(to_embed)} tasks, {total_keys} total NPZ keys.")
    print(f"Skipped {len(already_done)} already done.")
    print(f"\nAll embeddings in: {OUT_EMB}")


if __name__ == "__main__":
    main()
