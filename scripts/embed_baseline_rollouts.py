#!/usr/bin/env python3
"""Embed official baseline GRPO rollouts for early stopping simulation.

Reads JSONL rollout files, extracts action/response text per turn,
embeds via OpenAI text-embedding-3-large, and saves per-task NPZ files.
Also computes per-turn cumulative prefill token counts using the Qwen3 tokenizer.

Usage:
    python src/experiments/embed_baseline_rollouts.py --seed 1
    python src/experiments/embed_baseline_rollouts.py --seed 1 --seed 2 --seed 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path

# Load .env
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import numpy as np
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from _paths import RUNS, GRPO_REPLAY  # noqa: E402
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
MAX_CONCURRENT = 20
MAX_CHARS = 24000


def _parse_assistant_message(content: str) -> tuple[str, bool]:
    """Extract action text from a saved assistant message.

    Returns (action_text, is_tool_call).
    """
    if not content:
        return "<empty>", False

    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)
    if tc_match:
        try:
            tc = json.loads(tc_match.group(1))
            name = tc.get("name", "unknown")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            action = f"{name}({json.dumps(args, sort_keys=True)})"
            prefix = content[:tc_match.start()].strip()
            if prefix:
                action = f"{prefix}\n{action}"
            return action, True
        except json.JSONDecodeError:
            pass

    return content.strip() or "<empty>", False


def _parse_user_message(content: str) -> tuple[str, bool]:
    """Extract response text from a saved user message.

    Returns (response_text, is_tool_response).
    """
    if not content:
        return "<empty>", False

    tr_match = re.search(
        r"<tool_response>\s*(?:tool:\s*)?(.*?)\s*</tool_response>",
        content, re.DOTALL,
    )
    if tr_match:
        return f"[tool] {tr_match.group(1).strip()}", True

    clean = content.strip()
    if clean.startswith("user:"):
        clean = clean[5:].strip()
    return f"[user] {clean}" if clean else "<empty>", False


def extract_turns(conversation: list[dict]) -> list[dict]:
    """Extract (action_text, response_text) pairs from a conversation.

    Each turn = one assistant message + the following user message.
    Skips the system prompt and initial user greeting.
    """
    turns = []
    i = 0
    while i < len(conversation):
        msg = conversation[i]
        if msg["role"] == "assistant":
            action_text, is_tool_call = _parse_assistant_message(msg.get("content", ""))
            response_text = "<empty>"
            is_tool_response = False
            if i + 1 < len(conversation) and conversation[i + 1]["role"] == "user":
                response_text, is_tool_response = _parse_user_message(
                    conversation[i + 1].get("content", "")
                )
                i += 2
            else:
                i += 1
            turns.append({
                "action_text": action_text,
                "response_text": response_text,
                "is_tool_call": is_tool_call,
                "is_tool_response": is_tool_response,
            })
        else:
            i += 1
    return turns


def compute_per_turn_prefill(conversation: list[dict], tokenizer) -> list[int]:
    """Compute cumulative prefill token count at each turn boundary.

    Uses tokenizer.apply_chat_template to get exact token counts
    including system prompt, chat template tokens, etc.
    """
    assistant_indices = [
        i for i, m in enumerate(conversation) if m["role"] == "assistant"
    ]

    prefill_tokens = []
    for idx in assistant_indices:
        messages_so_far = conversation[:idx]
        try:
            rendered = tokenizer.apply_chat_template(
                messages_so_far, add_generation_prompt=True, tokenize=False
            )
            tokens = tokenizer.encode(rendered)
            prefill_tokens.append(len(tokens))
        except Exception:
            if prefill_tokens:
                prefill_tokens.append(prefill_tokens[-1] + 500)
            else:
                prefill_tokens.append(500)

    return prefill_tokens


async def embed_texts(
    client: AsyncOpenAI,
    texts: list[str],
    sem: asyncio.Semaphore,
    max_retries: int = 3,
) -> list[np.ndarray]:
    """Embed a list of texts, returning list of embeddings."""
    truncated = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]

    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.embeddings.create(
                    model=EMBED_MODEL, input=truncated
                )
            return [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise


async def embed_seed(seed: int, tokenizer):
    """Embed all rollouts for one seed."""
    log_dir = RUNS / f"tau2_baseline_retail_official_s{seed}" / "rollouts"
    out_dir = GRPO_REPLAY / f"official_retail_s{seed}"

    if not log_dir.exists():
        print(f"Seed {seed}: no rollout data at {log_dir}")
        return

    emb_act_dir = out_dir / "embeddings" / "action"
    emb_res_dir = out_dir / "embeddings" / "response"
    data_dir = out_dir / "data"
    emb_act_dir.mkdir(parents=True, exist_ok=True)
    emb_res_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(timeout=60.0)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    batch_files = sorted(log_dir.glob("batch_*.jsonl"))
    print(f"Seed {seed}: {len(batch_files)} batch files")

    total_rollouts = 0
    total_steps = 0
    total_texts = 0

    for bf in batch_files:
        batch_idx = int(bf.stem.split("_")[1])
        batch_name = f"batch_{batch_idx:02d}"

        # Group rollouts by task
        task_rollouts: dict[str, list[dict]] = defaultdict(list)
        with open(bf) as f:
            for line in f:
                r = json.loads(line)
                task_id = str(r["task_id"])
                task_rollouts[task_id].append(r)

        for task_id, rollouts in task_rollouts.items():
            task_label = f"{batch_name}/retail/{task_id}"
            stem = f"{batch_name}_retail_{task_id}"

            act_path = emb_act_dir / f"{stem}.npz"
            res_path = emb_res_dir / f"{stem}.npz"
            if act_path.exists() and res_path.exists():
                continue

            act_arrays = {}
            res_arrays = {}
            rewards = {}
            prefill_data = {}

            # Collect all texts to embed in one batch
            all_texts = []
            text_map = []  # (rollout_idx, step_idx, "act"|"res")

            for rollout in rollouts:
                rep = rollout["rep"]
                conv = rollout.get("conversation", [])
                turns = extract_turns(conv)

                if not turns:
                    continue

                prefill = compute_per_turn_prefill(conv, tokenizer)
                prefill_data[rep] = prefill
                rewards[rep] = rollout.get("reward", 0.0)

                for step_idx, turn in enumerate(turns):
                    all_texts.append(turn["action_text"])
                    text_map.append((rep, step_idx, "act"))
                    all_texts.append(turn["response_text"])
                    text_map.append((rep, step_idx, "res"))

            if not all_texts:
                continue

            # Embed in chunks of 100
            all_embeddings = []
            for chunk_start in range(0, len(all_texts), 100):
                chunk = all_texts[chunk_start:chunk_start + 100]
                embs = await embed_texts(client, chunk, sem)
                all_embeddings.extend(embs)

            # Organize into per-rollout arrays
            for (rep, step_idx, kind), emb in zip(text_map, all_embeddings):
                key = f"r{rep}_s{step_idx}"
                if kind == "act":
                    act_arrays[key] = emb
                else:
                    res_arrays[key] = emb

            np.savez_compressed(str(act_path), **act_arrays)
            np.savez_compressed(str(res_path), **res_arrays)

            # Save rewards and prefill data
            meta_path = data_dir / f"{stem}.json"
            meta = {
                "task_id": task_id,
                "batch": batch_name,
                "rewards": rewards,
                "prefill_tokens": prefill_data,
                "n_rollouts": len(rollouts),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, default=str)

            n_rollouts = len(rollouts)
            n_steps = len(all_texts) // 2
            total_rollouts += n_rollouts
            total_steps += n_steps
            total_texts += len(all_texts)

        print(
            f"  batch {batch_idx:02d}: {total_rollouts} rollouts, "
            f"{total_steps} steps, {total_texts} texts embedded"
        )

    print(
        f"Seed {seed} complete: {total_rollouts} rollouts, "
        f"{total_steps} steps, {total_texts} texts"
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    print(f"Loaded Qwen3-8B tokenizer")

    for seed in args.seed:
        await embed_seed(seed, tokenizer)


if __name__ == "__main__":
    asyncio.run(main())
