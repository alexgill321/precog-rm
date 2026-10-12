#!/usr/bin/env python3
"""OpenAI Batch Embedding Pipeline for TDP trajectories.

Extracts texts from Tau2-Bench TDP files, submits them to the OpenAI Batch
API for embedding, and saves structured embeddings as NPZ files
(per task / rollout / step / sample).

Three phases (resumable):
  1. Extract  — scan TDP files, extract & deduplicate texts, write JSONL
  2. Submit   — upload JSONL to OpenAI Batch API
  3. Collect  — poll for completion, download, save NPZ embeddings

Usage:
  python embed_openai.py \\
    --data-dir exp/tau2/baseline/multi_task_100/data \\
    --text-mode action

  python embed_openai.py \\
    --data-dir exp/tau2/baseline/multi_task_100/data \\
    --text-mode thinking

  python embed_openai.py --resume \\
    --data-dir exp/tau2/baseline/multi_task_100/data \\
    --text-mode thinking
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BATCH_REQUESTS = 30_000
MAX_JSONL_BYTES = 190 * 1024 * 1024  # 190 MB buffer under 200 MB API limit

EMBED_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}
PRICE_PER_M_TOKENS = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}

# ---------------------------------------------------------------------------
# Text extraction  (mirrors deprecated/code/tau2_bench_uq/uq_adapter.py, self-contained)
# ---------------------------------------------------------------------------


def _serialize_tool_calls(tool_calls: list[dict]) -> str:
    parts: list[str] = []
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        args = tc.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass
        parts.append(f"{name}({json.dumps(args, sort_keys=True)})")
    return " | ".join(parts)


def _sample_to_action_text(sample: dict) -> str:
    content = (sample.get("content") or "").strip()
    tool_calls = sample.get("tool_calls")
    parts: list[str] = []
    if content:
        parts.append(content)
    if tool_calls:
        parts.append(_serialize_tool_calls(tool_calls))
    return "\n".join(parts) if parts else "<empty>"


def extract_sample_text(sample: dict, text_mode: str) -> str:
    """Extract the text to embed from a raw sample dict."""
    if text_mode == "thinking":
        return (sample.get("reasoning_content") or "").strip() or "<empty>"

    action = _sample_to_action_text(sample)

    if text_mode == "full":
        reasoning = (sample.get("reasoning_content") or "").strip()
        parts: list[str] = []
        if reasoning:
            parts.append(reasoning)
        if action and action != "<empty>":
            parts.append(action)
        return "\n".join(parts) if parts else "<empty>"

    return action  # text_mode == "action"


def _input_message_to_text(input_message: dict | str | None) -> str:
    """Serialize a step's input_message to embeddable text."""
    if input_message is None:
        return "<empty>"
    if isinstance(input_message, str):
        return input_message.strip() or "<empty>"
    content = input_message.get("content", "")
    role = input_message.get("role", "")
    if isinstance(content, (dict, list)):
        content = json.dumps(content, sort_keys=True)
    content = str(content).strip()
    if not content:
        return "<empty>"
    if role:
        return f"[{role}] {content}"
    return content


def extract_response_text(
    steps: list[dict],
    step_idx: int,
    messages: list[dict] | None = None,
) -> str:
    """Extract the response to step_idx's action.

    Primary source: ``steps[step_idx+1].input_message``.
    Fallback for the last step: walk backward from the end of *messages*
    and return the last user/tool message (the environment's reply to the
    agent's final action).
    """
    next_idx = step_idx + 1
    if next_idx < len(steps):
        return _input_message_to_text(steps[next_idx].get("input_message"))
    if messages:
        for msg in reversed(messages):
            if msg.get("role") in ("user", "tool"):
                return _input_message_to_text(msg)
            if msg.get("role") == "assistant":
                break
    return "<empty>"


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

_RE_TASK_ID = re.compile(r'"task_id"\s*:\s*"([^"]*)"')
_RE_DOMAIN = re.compile(r'"domain"\s*:\s*"([^"]*)"')
_RE_TDP_INDEX = re.compile(r'"tdp_index"\s*:\s*(\d+)')
_RE_TDP_FILE_NUM = re.compile(r"_tdp_(\d+)\.json$")

_SKIP_DIRS = ("/embeddings/", "/uq_", "/checkpoints/", "/analysis/", "/plots/")


def parse_tdp_file_number(path: str) -> int:
    """Extract the NNN file number from a ``*_tdp_NNN.json`` filename."""
    m = _RE_TDP_FILE_NUM.search(path)
    if m is None:
        raise ValueError(f"Cannot parse file number from: {path}")
    return int(m.group(1))


def scan_tdp_files(data_dir: str) -> dict[str, list[tuple[int, str]]]:
    """Return ``{task_id: [(file_number, filepath), ...]}`` sorted by file number.

    The file number is extracted from the ``_tdp_NNN.json`` filename, which is
    the canonical rollout identifier across the pipeline.  (Previously this
    function sorted by ``tdp_index`` from inside the JSON, which diverged from
    the manifest for merged datasets with duplicate ``tdp_index`` values.)
    """
    patterns = [
        os.path.join(data_dir, "**", "*_tdp_*.json"),
        os.path.join(data_dir, "*_tdp_*.json"),
    ]
    files: set[str] = set()
    for pat in patterns:
        files.update(glob(pat, recursive=True))

    files_filtered = [
        f for f in sorted(files)
        if not any(d in f for d in _SKIP_DIRS)
    ]

    task_map: dict[str, list[tuple[int, str]]] = {}
    for fp in files_filtered:
        with open(fp) as fh:
            head = fh.read(2048)

        m_tid = _RE_TASK_ID.search(head)
        m_dom = _RE_DOMAIN.search(head)

        raw_id = m_tid.group(1) if m_tid else ""
        domain = m_dom.group(1) if m_dom else ""
        file_num = parse_tdp_file_number(fp)

        task_id = f"{domain}/{raw_id}" if domain else raw_id
        task_map.setdefault(task_id, []).append((file_num, fp))

    for tid in task_map:
        task_map[tid].sort(key=lambda x: x[0])

    return task_map


def task_id_to_filename(task_id: str) -> str:
    """Sanitise a task_id for use as a filename stem."""
    safe = task_id.replace("/", "_")
    safe = re.sub(r'[\\:*?"<>|]', "_", safe)
    if len(safe) > 200:
        h = hashlib.sha256(task_id.encode()).hexdigest()[:8]
        safe = safe[:190] + f"__{h}"
    return safe


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _state_path(output_dir: str) -> Path:
    return Path(output_dir) / "_batch_state.json"


def load_state(output_dir: str) -> dict | None:
    p = _state_path(output_dir)
    if p.exists():
        with open(p) as fh:
            return json.load(fh)
    return None


def save_state(output_dir: str, state: dict) -> None:
    p = _state_path(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    tmp.replace(p)
    print(f"  State saved → {p}")


# ---------------------------------------------------------------------------
# Phase 1: Extract
# ---------------------------------------------------------------------------


def phase_extract(
    data_dir: str,
    text_mode: str,
    model: str,
    output_dir: str,
) -> dict:
    """Scan TDP files, extract texts, deduplicate, write JSONL batch input."""
    print("\n═══ Phase 1: Extract ═══")

    print(f"  Scanning {data_dir} ...")
    task_map = scan_tdp_files(data_dir)
    n_tasks = len(task_map)
    n_files = sum(len(v) for v in task_map.values())
    print(f"  Found {n_files} TDP files across {n_tasks} tasks")

    text_to_idx: dict[str, int] = {}
    unique_texts: list[str] = []

    # structure[task_id] = [(file_num, rollout_steps), ...]
    #   rollout_steps = [step0_indices, step1_indices, ...]
    #   step_indices = [idx_sample0, idx_sample1, ...]
    structure: dict[str, list[tuple[int, list[list[int]]]]] = {}

    total_texts = 0
    skipped_empty = 0

    for task_i, (task_id, rollout_files) in enumerate(sorted(task_map.items())):
        task_rollouts: list[tuple[int, list[list[int]]]] = []

        for file_num, fp in rollout_files:
            with open(fp) as fh:
                raw = json.load(fh)

            rollout_steps: list[list[int]] = []
            all_steps = raw.get("steps", [])
            for step_i, step in enumerate(all_steps):
                step_indices: list[int] = []
                if text_mode == "response":
                    text = extract_response_text(all_steps, step_i, messages=raw.get("messages"))
                    total_texts += 1
                    if text == "<empty>":
                        skipped_empty += 1
                    if text not in text_to_idx:
                        text_to_idx[text] = len(unique_texts)
                        unique_texts.append(text)
                    step_indices.append(text_to_idx[text])
                else:
                    for sample in step.get("samples", []):
                        text = extract_sample_text(sample, text_mode)
                        total_texts += 1
                        if text == "<empty>":
                            skipped_empty += 1
                        if text not in text_to_idx:
                            text_to_idx[text] = len(unique_texts)
                            unique_texts.append(text)
                        step_indices.append(text_to_idx[text])
                rollout_steps.append(step_indices)

            task_rollouts.append((file_num, rollout_steps))
            del raw

        structure[task_id] = task_rollouts

        if (task_i + 1) % 10 == 0 or task_i + 1 == n_tasks:
            print(
                f"  Extracted {task_i + 1}/{n_tasks} tasks "
                f"({total_texts:,} total, {len(unique_texts):,} unique)"
            )

    total_chars = sum(len(t) for t in unique_texts)
    est_tokens = total_chars // 4
    embed_dim = EMBED_DIMS.get(model, 3072)
    price_per_m = PRICE_PER_M_TOKENS.get(model, 0.13)
    est_cost = (est_tokens / 1_000_000) * price_per_m * 0.5  # batch = 50% off

    print(f"\n  ── Summary ──")
    print(f"    Tasks:         {n_tasks}")
    print(f"    TDP files:     {n_files}")
    print(f"    Total texts:   {total_texts:,}")
    print(f"    Unique texts:  {len(unique_texts):,}  "
          f"(dedup saved {total_texts - len(unique_texts):,})")
    print(f"    Empty texts:   {skipped_empty:,}  (get zero vector)")
    print(f"    Est. tokens:   {est_tokens:,}")
    print(f"    Est. cost:     ${est_cost:.2f}  (batch pricing, 50% off)")

    # ── Write JSONL ──
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddable = [
        (idx, text)
        for idx, text in enumerate(unique_texts)
        if text != "<empty>"
    ]

    jsonl_files: list[str] = []
    current_lines: list[str] = []
    current_bytes = 0
    batch_i = 0

    def _flush() -> None:
        nonlocal current_lines, current_bytes, batch_i
        if not current_lines:
            return
        jsonl_path = out_dir / f"batch_{batch_i}.jsonl"
        with open(jsonl_path, "w") as fh:
            fh.writelines(current_lines)
        size_mb = jsonl_path.stat().st_size / (1024**2)
        jsonl_files.append(str(jsonl_path))
        print(f"  Wrote {jsonl_path.name}: {len(current_lines):,} requests ({size_mb:.1f} MB)")
        current_lines = []
        current_bytes = 0
        batch_i += 1

    for idx, text in embeddable:
        line = json.dumps({
            "custom_id": f"emb_{idx}",
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {"model": model, "input": text},
        }) + "\n"
        line_bytes = len(line.encode("utf-8"))

        needs_split = (
            len(current_lines) >= MAX_BATCH_REQUESTS
            or (current_bytes + line_bytes > MAX_JSONL_BYTES and current_lines)
        )
        if needs_split:
            _flush()

        current_lines.append(line)
        current_bytes += line_bytes

    _flush()

    state = {
        "config": {
            "model": model,
            "text_mode": text_mode,
            "data_dir": str(Path(data_dir).resolve()),
            "embed_dim": embed_dim,
        },
        "stats": {
            "n_tasks": n_tasks,
            "n_files": n_files,
            "total_texts": total_texts,
            "unique_texts": len(unique_texts),
            "embeddable_texts": len(embeddable),
            "est_tokens": est_tokens,
            "est_cost_usd": round(est_cost, 4),
        },
        "phase": "extracted",
        "jsonl_files": jsonl_files,
        "batch_ids": [],
        "empty_idx": text_to_idx.get("<empty>"),
        "structure": structure,
    }
    save_state(output_dir, state)
    return state


# ---------------------------------------------------------------------------
# Phase 2: Submit
# ---------------------------------------------------------------------------


def phase_submit(output_dir: str) -> dict:
    """Upload JSONL file(s) to OpenAI and create batch job(s)."""
    print("\n═══ Phase 2: Submit ═══")

    state = load_state(output_dir)
    if state is None:
        raise RuntimeError("No state file found. Run extract phase first.")

    if state["phase"] != "extracted":
        print(f"  Phase is '{state['phase']}', skipping submit.")
        return state

    from openai import OpenAI

    client = OpenAI()

    batch_ids: list[str] = []
    for jsonl_path in state["jsonl_files"]:
        print(f"  Uploading {jsonl_path} ...")
        with open(jsonl_path, "rb") as fh:
            input_file = client.files.create(file=fh, purpose="batch")
        print(f"    File ID: {input_file.id}")

        batch = client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
            metadata={
                "description": (
                    f"TDP embeddings — {state['config']['text_mode']} — "
                    f"{state['stats']['embeddable_texts']} texts"
                ),
            },
        )
        batch_ids.append(batch.id)
        print(f"    Batch ID: {batch.id}")

    state["batch_ids"] = batch_ids
    state["phase"] = "submitted"
    save_state(output_dir, state)

    print(f"\n  {len(batch_ids)} batch(es) submitted. "
          f"Use --resume to poll & collect when ready.")
    return state


# ---------------------------------------------------------------------------
# Phase 3: Collect
# ---------------------------------------------------------------------------


def phase_collect(output_dir: str, poll_interval: float = 30.0) -> dict:
    """Poll batch status, download results, save NPZ embeddings."""
    print("\n═══ Phase 3: Collect ═══")

    state = load_state(output_dir)
    if state is None:
        raise RuntimeError("No state file found.")

    if state["phase"] == "completed":
        print("  Already completed.")
        return state

    if state["phase"] != "submitted":
        raise RuntimeError(f"Expected phase 'submitted', got '{state['phase']}'")

    from openai import OpenAI

    client = OpenAI()

    embed_dim: int = state["config"]["embed_dim"]
    model: str = state["config"]["model"]

    # ── Poll & download all batches ──
    all_embeddings: dict[int, np.ndarray] = {}

    for batch_id in state["batch_ids"]:
        print(f"  Polling batch {batch_id} ...")
        batch = client.batches.retrieve(batch_id)

        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            counts = batch.request_counts
            total = counts.total if counts else "?"
            done = counts.completed if counts else "?"
            fail = counts.failed if counts else 0
            print(
                f"    {batch.status}: {done}/{total} done, "
                f"{fail} failed  (next poll in {poll_interval:.0f}s)"
            )
            time.sleep(poll_interval)
            batch = client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {batch_id} ended with status: {batch.status}"
            )

        print(f"  Batch {batch_id} completed — downloading results ...")
        output_content = client.files.content(batch.output_file_id)

        n_parsed = 0
        n_errors = 0
        for line in output_content.text.strip().split("\n"):
            if not line:
                continue
            result = json.loads(line)
            cid: str = result["custom_id"]
            idx = int(cid.split("_", 1)[1])

            if result.get("error"):
                n_errors += 1
                all_embeddings[idx] = np.zeros(embed_dim, dtype=np.float32)
                continue

            body = result["response"]["body"]
            embedding = body["data"][0]["embedding"]
            all_embeddings[idx] = np.array(embedding, dtype=np.float32)
            n_parsed += 1

        print(f"    Parsed {n_parsed:,} embeddings, {n_errors} errors")

    # ── Build NPZ files per task ──
    out_dir = Path(output_dir)
    structure: dict[str, list] = state["structure"]
    empty_idx = state.get("empty_idx")
    zero_vec = np.zeros(embed_dim, dtype=np.float32)

    print(f"\n  Saving NPZ files ({len(structure)} tasks) ...")

    task_stats: dict[str, dict] = {}
    for task_i, (task_id, rollouts) in enumerate(sorted(structure.items())):
        arrays: dict[str, np.ndarray] = {}

        for file_num, steps in rollouts:
            for s_idx, sample_indices in enumerate(steps):
                key = f"r{file_num}_s{s_idx}"
                vecs = []
                for text_idx in sample_indices:
                    if text_idx == empty_idx:
                        vecs.append(zero_vec)
                    else:
                        vecs.append(all_embeddings.get(text_idx, zero_vec))
                if vecs:
                    arrays[key] = np.stack(vecs)
                else:
                    arrays[key] = np.zeros((0, embed_dim), dtype=np.float32)

        fname = task_id_to_filename(task_id) + ".npz"
        np.savez_compressed(out_dir / fname, **arrays)

        n_rollouts = len(rollouts)
        n_steps = sum(len(s) for _, s in rollouts)
        n_samples = sum(
            len(samples) for _, steps in rollouts for samples in steps
        )
        task_stats[task_id] = {
            "n_rollouts": n_rollouts,
            "n_steps": n_steps,
            "n_samples": n_samples,
            "file": fname,
        }

        if (task_i + 1) % 20 == 0 or task_i + 1 == len(structure):
            print(f"    Saved {task_i + 1}/{len(structure)} tasks")

    # ── Write manifest ──
    manifest = {
        "model": model,
        "embed_dim": embed_dim,
        "text_mode": state["config"]["text_mode"],
        "data_dir": state["config"]["data_dir"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": state["stats"],
        "tasks": task_stats,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n  Manifest → {manifest_path}")

    state["phase"] = "completed"
    save_state(output_dir, state)

    total_npz = len(structure)
    total_samples = sum(t["n_samples"] for t in task_stats.values())
    print(f"\n  ✓ Done! {total_npz} NPZ files, {total_samples:,} sample embeddings")
    print(f"    Output: {out_dir}")
    return state


# ---------------------------------------------------------------------------
# Synchronous embedding (fallback when Batch API is unavailable)
# ---------------------------------------------------------------------------

SYNC_CHUNK_SIZE = 128  # max texts per API call (thinking texts can be very long)


def phase_sync_embed(output_dir: str, max_workers: int = 8) -> dict:
    """Embed all texts via synchronous API calls (parallel chunks)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n═══ Sync Embed ═══")

    state = load_state(output_dir)
    if state is None:
        raise RuntimeError("No state file found. Run extract phase first.")

    if state["phase"] == "completed":
        print("  Already completed.")
        return state

    from openai import OpenAI
    client = OpenAI()

    embed_dim: int = state["config"]["embed_dim"]
    model: str = state["config"]["model"]

    # Build list of (index, text) to embed from the saved JSONL files
    embeddable: list[tuple[int, str]] = []
    for jsonl_path in state["jsonl_files"]:
        with open(jsonl_path) as fh:
            for line in fh:
                obj = json.loads(line)
                idx = int(obj["custom_id"].split("_", 1)[1])
                text = obj["body"]["input"]
                embeddable.append((idx, text))

    n_total = len(embeddable)
    print(f"  {n_total:,} texts to embed  (model: {model})")

    # Split into chunks
    chunks: list[list[tuple[int, str]]] = []
    for i in range(0, n_total, SYNC_CHUNK_SIZE):
        chunks.append(embeddable[i : i + SYNC_CHUNK_SIZE])
    print(f"  {len(chunks)} API calls  ({SYNC_CHUNK_SIZE} texts/call, {max_workers} workers)")

    all_embeddings: dict[int, np.ndarray] = {}
    n_done = 0
    n_errors = 0

    def _embed_chunk(chunk: list[tuple[int, str]]) -> list[tuple[int, np.ndarray]]:
        indices = [idx for idx, _ in chunk]
        texts = [text for _, text in chunk]
        resp = client.embeddings.create(model=model, input=texts)
        results = []
        for item in resp.data:
            results.append((indices[item.index], np.array(item.embedding, dtype=np.float32)))
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_embed_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            chunk_i = futures[future]
            try:
                results = future.result()
                for idx, vec in results:
                    all_embeddings[idx] = vec
                n_done += len(results)
            except Exception as e:
                chunk = chunks[chunk_i]
                n_errors += len(chunk)
                print(f"    ERROR chunk {chunk_i}: {e}")
                zero = np.zeros(embed_dim, dtype=np.float32)
                for idx, _ in chunk:
                    all_embeddings[idx] = zero

            if (n_done + n_errors) % (SYNC_CHUNK_SIZE * 4) < SYNC_CHUNK_SIZE or n_done + n_errors == n_total:
                print(f"    {n_done + n_errors:,}/{n_total:,} done  ({n_errors} errors)")

    print(f"  Embedded {n_done:,} texts, {n_errors} errors")

    # Build NPZ files (same logic as phase_collect)
    out_dir = Path(output_dir)
    structure: dict[str, list] = state["structure"]
    empty_idx = state.get("empty_idx")
    zero_vec = np.zeros(embed_dim, dtype=np.float32)

    print(f"\n  Saving NPZ files ({len(structure)} tasks) ...")

    task_stats: dict[str, dict] = {}
    for task_i, (task_id, rollouts) in enumerate(sorted(structure.items())):
        arrays: dict[str, np.ndarray] = {}

        for file_num, steps in rollouts:
            for s_idx, sample_indices in enumerate(steps):
                key = f"r{file_num}_s{s_idx}"
                vecs = []
                for text_idx in sample_indices:
                    if text_idx == empty_idx:
                        vecs.append(zero_vec)
                    else:
                        vecs.append(all_embeddings.get(text_idx, zero_vec))
                if vecs:
                    arrays[key] = np.stack(vecs)
                else:
                    arrays[key] = np.zeros((0, embed_dim), dtype=np.float32)

        fname = task_id_to_filename(task_id) + ".npz"
        np.savez_compressed(out_dir / fname, **arrays)

        n_rollouts = len(rollouts)
        n_steps = sum(len(s) for _, s in rollouts)
        n_samples = sum(
            len(samples) for _, steps in rollouts for samples in steps
        )
        task_stats[task_id] = {
            "n_rollouts": n_rollouts,
            "n_steps": n_steps,
            "n_samples": n_samples,
            "file": fname,
        }

        if (task_i + 1) % 20 == 0 or task_i + 1 == len(structure):
            print(f"    Saved {task_i + 1}/{len(structure)} tasks")

    manifest = {
        "model": model,
        "embed_dim": embed_dim,
        "text_mode": state["config"]["text_mode"],
        "data_dir": state["config"]["data_dir"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": state["stats"],
        "tasks": task_stats,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n  Manifest → {manifest_path}")

    state["phase"] = "completed"
    save_state(output_dir, state)

    total_npz = len(structure)
    total_samples = sum(t["n_samples"] for t in task_stats.values())
    print(f"\n  ✓ Done! {total_npz} NPZ files, {total_samples:,} sample embeddings")
    print(f"    Output: {out_dir}")
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI Batch Embedding Pipeline for TDP trajectories",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to experiment directory containing TDP files",
    )
    parser.add_argument(
        "--text-mode",
        choices=["action", "thinking", "full", "response"],
        default="action",
        help="Which text to embed (default: action)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-large",
        help="OpenAI embedding model (default: text-embedding-3-large)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: {data-dir}/embeddings/{text-mode})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved batch state instead of starting fresh",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run extract phase only — show stats, write JSONL, don't submit",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous API calls instead of Batch API",
    )
    parser.add_argument(
        "--sync-workers",
        type=int,
        default=8,
        help="Number of parallel workers for --sync mode (default: 8)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=30.0,
        help="Seconds between batch status polls (default: 30)",
    )

    args = parser.parse_args()

    # Load .env for OPENAI_API_KEY
    try:
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(_env_path)
    except ImportError:
        pass

    output_dir = args.output_dir or os.path.join(
        args.data_dir, "embeddings", args.text_mode,
    )

    print("OpenAI Batch Embedding Pipeline")
    print(f"  data_dir:   {args.data_dir}")
    print(f"  text_mode:  {args.text_mode}")
    print(f"  model:      {args.model}")
    print(f"  output_dir: {output_dir}")
    if args.sync:
        print(f"  mode:       sync ({args.sync_workers} workers)")
    elif args.resume:
        print(f"  mode:       resume")
    elif args.dry_run:
        print(f"  mode:       dry-run")
    else:
        print(f"  mode:       full run")

    # Determine starting phase
    if args.resume:
        state = load_state(output_dir)
        if state is None:
            print("\n  No state file found — starting fresh.")
            phase = ""
        else:
            phase = state.get("phase", "")
            print(f"\n  Resuming from phase: {phase}")
    else:
        phase = ""

    # Phase 1: Extract
    if phase in ("", ):
        state = phase_extract(
            args.data_dir, args.text_mode, args.model, output_dir,
        )
        phase = "extracted"

    if args.dry_run:
        print("\n  Dry run complete. Review stats above, then run without "
              "--dry-run (use --resume to skip re-extraction).")
        return

    if args.sync:
        phase_sync_embed(output_dir, max_workers=args.sync_workers)
        return

    # Phase 2: Submit
    if phase == "extracted":
        state = phase_submit(output_dir)
        phase = state["phase"]

    # Phase 3: Collect
    if phase == "submitted":
        phase_collect(output_dir, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
