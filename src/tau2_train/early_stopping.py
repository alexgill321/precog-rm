"""ProtoNetPRM early stopping for GRPO rollouts.

Integrates prefix-based early stopping into the rollout collection pipeline.
At each turn, embeds the agent action + environment response via OpenAI,
scores the prefix with ProtoNetPRM, and terminates early if confident.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from tau2_train.config import CostTracker, EarlyStopConfig
from tau2_train.renderer import Renderer, Message
from tau2_train.rollout import (
    _build_system_prompt,
    _format_action_for_gym,
    _extract_structured_data,
    _recompute_reward_with_action,
)
from tau2_train.types import (
    TokensWithLogprobs,
    Trajectory,
    TrajectoryGroup,
    TrajectoryID,
    Transition,
)

logger = logging.getLogger(__name__)

EMBED_DIM = 3072


# ── Text extraction (bridges live rollout data to embedding format) ──


def _action_text_from_parsed(parsed_message: dict) -> str:
    """Extract action text from a live parsed agent message.

    Same logic as src/prm/embeddings._sample_to_action_text.
    """
    content = (parsed_message.get("content") or "").strip()
    tool_calls = parsed_message.get("tool_calls")
    parts: list[str] = []
    if content:
        parts.append(content)
    if tool_calls:
        tc_parts = []
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    pass
            tc_parts.append(f"{name}({json.dumps(args, sort_keys=True)})")
        parts.append(" | ".join(tc_parts))
    return "\n".join(parts) if parts else "<empty>"


def _response_text_from_obs(obs_str: str, is_tool_response: bool) -> str:
    """Format environment observation as embeddable text.

    Same format as src/prm/embeddings._input_message_to_text.
    """
    if not obs_str or not obs_str.strip():
        return "<empty>"
    role = "tool" if is_tool_response else "user"
    return f"[{role}] {obs_str.strip()}"


# ── Async embedding client ──────────────────────────────────────────


class EmbeddingClient:
    """Async wrapper around OpenAI embeddings API."""

    def __init__(self, model: str = "text-embedding-3-large", max_concurrent: int = 10):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(timeout=60.0)
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrent)
        self.total_texts = 0
        self.total_tokens = 0

    @staticmethod
    def _truncate(text: str, max_chars: int = 24000) -> str:
        return text[:max_chars] if len(text) > max_chars else text

    async def embed_step(
        self, action_text: str, response_text: str,
        max_retries: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed one step's action + response texts. Returns (act_emb, res_emb)."""
        action_text = self._truncate(action_text)
        response_text = self._truncate(response_text)
        for attempt in range(max_retries):
            try:
                async with self._sem:
                    resp = await self._client.embeddings.create(
                        model=self._model,
                        input=[action_text, response_text],
                    )
                self.total_texts += 2
                self.total_tokens += resp.usage.total_tokens
                act_emb = np.array(resp.data[0].embedding, dtype=np.float32)
                res_emb = np.array(resp.data[1].embedding, dtype=np.float32)
                return act_emb, res_emb
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    logger.warning(f"Embedding failed (attempt {attempt+1}): {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise

    @property
    def estimated_cost(self) -> float:
        return self.total_tokens / 1_000_000 * 0.13


# ── Support set manager ─────────────────────────────────────────────


class SupportSetManager:
    """Per-task support set state for growing-support early stopping.

    Supports staged additions: call stage_rollout() during a batch, then
    commit_staged() after the batch succeeds. If the batch crashes, staged
    entries are lost automatically.
    """

    def __init__(
        self,
        save_dir: str | None = None,
        rolling_window: int | None = None,
        min_class_support: int = 16,
    ):
        self._supports: dict[str, dict] = {}
        self._permanent: dict[str, dict] = {}
        self._staged: list[tuple[str, np.ndarray, np.ndarray, int, int | None]] = []
        self._save_dir = save_dir
        self._rolling_window = rolling_window
        self._min_class_support = min_class_support

    def has_viable_support(self, task_key: str) -> bool:
        _, _, labels = self._get_combined(task_key)
        return len(labels) >= 2 and len(set(labels)) >= 2

    def get_support(self, task_key: str) -> tuple[list, list, list]:
        return self._get_combined(task_key)

    def _get_combined(self, task_key: str) -> tuple[list, list, list]:
        """Merge permanent + rolling support for a task."""
        act, res, labels = [], [], []
        perm = self._permanent.get(task_key)
        if perm is not None:
            act = list(perm["act"])
            res = list(perm["res"])
            labels = list(perm["labels"])
        entry = self._supports.get(task_key)
        if entry is not None:
            act = act + entry["act"]
            res = res + entry["res"]
            labels = labels + entry["labels"]
        return act, res, labels

    def stage_rollout(
        self,
        task_key: str,
        act_seq: np.ndarray,
        res_seq: np.ndarray,
        label: int,
        batch_idx: int | None = None,
    ):
        self._staged.append((task_key, act_seq, res_seq, label, batch_idx))

    def commit_staged(self):
        for task_key, act_seq, res_seq, label, batch_idx in self._staged:
            self._add_rollout(task_key, act_seq, res_seq, label, batch_idx)
        n = len(self._staged)
        self._staged.clear()
        if self._rolling_window is not None:
            self._evict_rolling()
        return n

    def _evict_rolling(self):
        """Evict old batches per task, keeping the most recent rolling_window batches.

        Retains extra entries from older batches if needed to maintain
        min_class_support examples per class (pass/fail).
        """
        for task_key, entry in self._supports.items():
            batches = entry["batches"]
            labels = entry["labels"]

            distinct = sorted(set(b for b in batches if b is not None))
            if len(distinct) <= self._rolling_window:
                continue

            keep_batches = set(distinct[-self._rolling_window:])
            keep_idx = []
            evict_idx = []
            for i, b in enumerate(batches):
                if b is None or b in keep_batches:
                    keep_idx.append(i)
                else:
                    evict_idx.append(i)

            if not evict_idx:
                continue

            n_pass = sum(1 for i in keep_idx if labels[i] == 1)
            n_fail = sum(1 for i in keep_idx if labels[i] == 0)

            retain = set()
            if n_pass < self._min_class_support:
                candidates = [i for i in evict_idx if labels[i] == 1]
                candidates.sort(key=lambda i: batches[i] or 0, reverse=True)
                for i in candidates[: self._min_class_support - n_pass]:
                    retain.add(i)

            if n_fail < self._min_class_support:
                candidates = [i for i in evict_idx if labels[i] == 0]
                candidates.sort(key=lambda i: batches[i] or 0, reverse=True)
                for i in candidates[: self._min_class_support - n_fail]:
                    retain.add(i)

            final = sorted(set(keep_idx) | retain)
            if len(final) == len(labels):
                continue
            entry["act"] = [entry["act"][i] for i in final]
            entry["res"] = [entry["res"][i] for i in final]
            entry["labels"] = [entry["labels"][i] for i in final]
            entry["batches"] = [entry["batches"][i] for i in final]

    def discard_staged(self):
        n = len(self._staged)
        self._staged.clear()
        return n

    def _add_rollout(
        self,
        task_key: str,
        act_seq: np.ndarray,
        res_seq: np.ndarray,
        label: int,
        batch_idx: int | None = None,
    ):
        if task_key not in self._supports:
            self._supports[task_key] = {"act": [], "res": [], "labels": [], "batches": []}
        entry = self._supports[task_key]
        entry["act"].append(act_seq)
        entry["res"].append(res_seq)
        entry["labels"].append(label)
        entry.setdefault("batches", []).append(batch_idx)

    def drop_batches_after(self, max_batch: int):
        """Remove all rollouts added from batches > max_batch."""
        for task_key, entry in self._supports.items():
            batches = entry.get("batches", [])
            if not batches:
                continue
            keep = [i for i, b in enumerate(batches) if b is None or b <= max_batch]
            if len(keep) == len(batches):
                continue
            entry["act"] = [entry["act"][i] for i in keep]
            entry["res"] = [entry["res"][i] for i in keep]
            entry["labels"] = [entry["labels"][i] for i in keep]
            entry["batches"] = [entry["batches"][i] for i in keep]
        # Remove tasks with no rollouts left
        self._supports = {k: v for k, v in self._supports.items() if v["labels"]}

    def load_permanent_support(
        self,
        manifest_path: str | Path,
        embeddings_dir: str | Path,
        n_per_task: int = 16,
        seed: int = 42,
    ):
        """Load pre-computed embeddings as permanent (non-evictable) support.

        These are merged with rolling GRPO support in get_support() but
        never touched by rolling eviction.

        Args:
            manifest_path: JSON with task_label -> {rollouts: [{rid, reward, ...}]}
            embeddings_dir: directory with action/ and response/ subdirs of NPZ files
            n_per_task: max rollouts to sample per task
            seed: random seed for subsampling
        """
        import re as _re
        manifest_path = Path(manifest_path)
        embeddings_dir = Path(embeddings_dir)
        rng = np.random.RandomState(seed)

        with open(manifest_path) as f:
            manifest = json.load(f)

        loaded = 0
        for task_label, entry in manifest.items():
            if not task_label.startswith("retail/"):
                continue
            task_id = task_label.split("/")[1]
            task_key = f"retail_{task_id}"

            stem = task_label.replace("/", "_")
            act_path = embeddings_dir / "action" / f"{stem}.npz"
            res_path = embeddings_dir / "response" / f"{stem}.npz"
            if not act_path.exists() or not res_path.exists():
                continue

            act_npz = np.load(str(act_path))
            res_npz = np.load(str(res_path))

            rollout_data = []
            for r in entry["rollouts"]:
                rid = r["rid"]
                reward = r.get("reward", 0.0)
                label = 1 if (reward or 0) >= 0.5 else 0

                act_steps, res_steps = {}, {}
                for key in act_npz.files:
                    m = _re.match(rf"r{rid}_s(\d+)", key)
                    if m:
                        act_steps[int(m.group(1))] = act_npz[key].squeeze()
                for key in res_npz.files:
                    m = _re.match(rf"r{rid}_s(\d+)", key)
                    if m:
                        res_steps[int(m.group(1))] = res_npz[key].squeeze()

                steps = sorted(set(act_steps.keys()) & set(res_steps.keys()))
                if not steps:
                    continue
                act_seq = np.stack([act_steps[s] for s in steps])
                res_seq = np.stack([res_steps[s] for s in steps])
                rollout_data.append((act_seq, res_seq, label))

            if not rollout_data or len(set(rd[2] for rd in rollout_data)) < 2:
                continue

            if n_per_task < len(rollout_data):
                pass_idx = [i for i, rd in enumerate(rollout_data) if rd[2] == 1]
                fail_idx = [i for i, rd in enumerate(rollout_data) if rd[2] == 0]
                rng.shuffle(pass_idx)
                rng.shuffle(fail_idx)
                half = n_per_task // 2
                n_pass = min(half, len(pass_idx))
                n_fail = min(half, len(fail_idx))
                # Fill remaining slots from the other class
                if n_pass < half:
                    n_fail = min(n_per_task - n_pass, len(fail_idx))
                elif n_fail < half:
                    n_pass = min(n_per_task - n_fail, len(pass_idx))
                selected = pass_idx[:n_pass] + fail_idx[:n_fail]
            else:
                selected = range(len(rollout_data))

            self._permanent[task_key] = {
                "act": [rollout_data[i][0] for i in selected],
                "res": [rollout_data[i][1] for i in selected],
                "labels": [rollout_data[i][2] for i in selected],
            }
            loaded += 1

        logger.info(
            f"Loaded permanent support for {loaded} tasks "
            f"({n_per_task} rollouts/task) from {manifest_path.parent}"
        )

    def task_count(self) -> int:
        all_keys = set(self._supports.keys()) | set(self._permanent.keys())
        return len(all_keys)

    def viable_task_count(self) -> int:
        all_keys = set(self._supports.keys()) | set(self._permanent.keys())
        return sum(1 for k in all_keys if self.has_viable_support(k))

    def memory_bytes(self) -> int:
        total = 0
        for entry in self._supports.values():
            for a, r in zip(entry["act"], entry["res"]):
                total += a.nbytes + r.nbytes
        for entry in self._permanent.values():
            for a, r in zip(entry["act"], entry["res"]):
                total += a.nbytes + r.nbytes
        return total

    def summary(self) -> dict[str, int]:
        return {
            k: len(v["labels"])
            for k, v in sorted(self._supports.items())
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        index = {}
        for task_key, entry in self._supports.items():
            safe_name = task_key.replace("/", "_")
            npz_path = path / f"{safe_name}.npz"
            arrays = {}
            for i, (act, res) in enumerate(zip(entry["act"], entry["res"])):
                arrays[f"act_{i}"] = act
                arrays[f"res_{i}"] = res
            np.savez_compressed(str(npz_path), **arrays)
            index[task_key] = {
                "file": safe_name + ".npz",
                "labels": entry["labels"],
                "batches": entry.get("batches", []),
                "count": len(entry["labels"]),
            }

        with open(path / "index.json", "w") as f:
            json.dump(index, f, indent=2)

    def load(self, path: str | Path):
        path = Path(path)
        index_file = path / "index.json"
        if not index_file.exists():
            return

        with open(index_file) as f:
            index = json.load(f)

        skipped = 0
        for task_key, info in index.items():
            npz_path = path / info["file"]
            if not npz_path.exists():
                continue
            # Skip corrupt npz files (truncated mid-write by SIGTERM at wall
            # timeout); rolling support self-refills from subsequent rollouts,
            # permanent support is independent.
            try:
                data = np.load(str(npz_path))
                n = info["count"]
                act_list = [data[f"act_{i}"] for i in range(n)]
                res_list = [data[f"res_{i}"] for i in range(n)]
            except Exception as e:
                logger.warning(
                    f"Skipping corrupt support file {npz_path.name} for {task_key}: "
                    f"{type(e).__name__}: {e}"
                )
                skipped += 1
                continue
            self._supports[task_key] = {
                "act": act_list,
                "res": res_list,
                "labels": info["labels"],
                "batches": info.get("batches", [None] * n),
            }
        if skipped:
            logger.warning(f"Skipped {skipped} corrupt support files")
        logger.info(
            f"Loaded support sets for {len(self._supports)} tasks from {path}"
        )


# ── Modified rollout with early stopping ─────────────────────────────


async def rollout_episode_with_early_stopping(
    domain: str,
    task_id: str,
    sampling_client: Any,
    renderer: Renderer,
    max_tokens: int,
    temperature: float,
    max_turns: int,
    user_model: str,
    seed: int | None = None,
    cost_tracker: CostTracker | None = None,
    use_action_reward: bool = False,
    *,
    prm: Any | None = None,
    embed_client: EmbeddingClient | None = None,
    threshold: float = 0.8,
    min_steps: int = 1,
    can_early_stop: bool = False,
) -> tuple[list[Transition], float, list[dict], dict]:
    """Run a single rollout episode with optional ProtoNet early stopping.

    Always embeds each step (for support set construction).
    Only scores + early-stops when can_early_stop=True (viable support exists).
    """
    import tinker
    from tau2.gym import AgentGymEnv

    t_start = time.time()
    # Per-rollout phase timing accumulators (seconds, perf_counter).
    # Within a single rollout, ops are sequential, so these partition the
    # rollout's own timeline. score is synchronous (clean wall-clock); gen/
    # env/embed wrap awaits, so under concurrency they also absorb event-loop
    # resume delay (elapsed-with-contention, not pure CPU).
    gen_elapsed_s = 0.0
    env_elapsed_s = 0.0
    gym_env = AgentGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_turns,
        user_llm=user_model,
    )

    _env_t0 = time.perf_counter()
    obs_str, info = await asyncio.to_thread(gym_env.reset, seed)
    env_elapsed_s += time.perf_counter() - _env_t0
    policy = info.get("policy", "")
    tools = info.get("tools", [])

    system_prompt = _build_system_prompt(policy, tools)
    conversation: list[dict] = [{"role": "system", "content": system_prompt}]

    if obs_str:
        conversation.append({"role": "user", "content": obs_str})

    transitions: list[Transition] = []
    final_reward = 0.0
    original_reward = 0.0
    termination_reason = "max_turns"
    reward_info = None
    structured_messages: list[dict] = []
    structured_steps: list[dict] = []
    total_prefill = 0
    total_sample = 0

    # Early stopping state
    act_embeddings: list[np.ndarray] = []
    res_embeddings: list[np.ndarray] = []
    prm_scores: list[float] = []
    early_stopped = False
    early_stop_turn = None

    embed_wall_clock_s = 0.0
    score_wall_clock_s = 0.0

    messages_for_renderer = [
        Message(role=m["role"], content=m["content"]) for m in conversation
    ]
    model_input = renderer.build_generation_prompt(messages_for_renderer)
    running_tokens: list[int] = list(model_input.to_ints())

    for turn in range(max_turns):
        model_input = tinker.ModelInput.from_ints(running_tokens)

        input_token_count = model_input.length
        context_limit = 32000
        available = context_limit - input_token_count
        if available <= 0:
            logger.warning(
                f"Context full at turn {turn} ({input_token_count} tokens), "
                "ending episode"
            )
            termination_reason = "context_full"
            break
        effective_max_tokens = min(max_tokens, available)

        sampling_params = tinker.SamplingParams(
            stop=renderer.get_stop_sequences(),
            max_tokens=effective_max_tokens,
            temperature=temperature,
        )
        _gen_t0 = time.perf_counter()
        sample_result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        gen_elapsed_s += time.perf_counter() - _gen_t0

        sampled_seq = sample_result.sequences[0]
        response_tokens = sampled_seq.tokens
        response_logprobs = sampled_seq.logprobs

        total_prefill += input_token_count
        total_sample += len(response_tokens)
        if cost_tracker is not None:
            cost_tracker.prefill_tokens += input_token_count
            cost_tracker.sample_tokens += len(response_tokens)

        parsed_message, _parse_ok = renderer.parse_response(response_tokens)
        response_text = parsed_message.get("content", "")

        action_str = _format_action_for_gym(parsed_message)

        _env_t0 = time.perf_counter()
        obs_str, reward, terminated, truncated, step_info = await asyncio.to_thread(
            gym_env.step, action_str,
        )
        env_elapsed_s += time.perf_counter() - _env_t0

        transition = Transition(
            ob=model_input,
            ac=TokensWithLogprobs(
                tokens=response_tokens, maybe_logprobs=response_logprobs,
            ),
            reward=reward if terminated else 0.0,
            episode_done=terminated,
        )
        transitions.append(transition)

        conversation.append({"role": "assistant", "content": response_text})

        running_tokens.extend(response_tokens)

        stop_token = renderer.get_stop_sequences()[0]
        if not response_tokens or response_tokens[-1] != stop_token:
            running_tokens.append(stop_token)

        # --- Embed this step ---
        if embed_client is not None and not terminated:
            is_tool_response = bool(
                parsed_message.get("tool_calls")
            )
            action_text = _action_text_from_parsed(parsed_message)
            obs_text = _response_text_from_obs(obs_str or "", is_tool_response)
            try:
                _embed_t0 = time.perf_counter()
                act_emb, res_emb = await embed_client.embed_step(action_text, obs_text)
                embed_wall_clock_s += time.perf_counter() - _embed_t0
                act_embeddings.append(act_emb)
                res_embeddings.append(res_emb)
            except Exception as e:
                logger.warning(
                    f"Embedding failed at turn {turn} ({domain}/{task_id}): "
                    f"{type(e).__name__}: {e} | "
                    f"action_len={len(action_text)} response_len={len(obs_text)}"
                )

            # --- Score prefix and maybe early stop ---
            if (
                can_early_stop
                and prm is not None
                and len(act_embeddings) >= min_steps
            ):
                act_stack = np.stack(act_embeddings)
                res_stack = np.stack(res_embeddings)
                _score_t0 = time.perf_counter()
                score = prm.score_prefix(act_stack, res_stack)
                score_wall_clock_s += time.perf_counter() - _score_t0
                prm_scores.append(score)

                if score > threshold:
                    # Predict pass → terminate early
                    predicted_reward = 1.0
                    early_stopped = True
                    early_stop_turn = turn
                    final_reward = predicted_reward
                    transitions[-1] = Transition(
                        ob=transitions[-1].ob,
                        ac=transitions[-1].ac,
                        reward=predicted_reward,
                        episode_done=True,
                    )
                    termination_reason = "early_stop_pass"
                    logger.debug(
                        f"Early stop PASS at turn {turn} "
                        f"(score={score:.3f}, thresh={threshold}) "
                        f"({domain}/{task_id})"
                    )
                    break
                elif (1 - score) > threshold:
                    # Predict fail → terminate early
                    predicted_reward = 0.0
                    early_stopped = True
                    early_stop_turn = turn
                    final_reward = predicted_reward
                    transitions[-1] = Transition(
                        ob=transitions[-1].ob,
                        ac=transitions[-1].ac,
                        reward=predicted_reward,
                        episode_done=True,
                    )
                    termination_reason = "early_stop_fail"
                    logger.debug(
                        f"Early stop FAIL at turn {turn} "
                        f"(score={score:.3f}, thresh={threshold}) "
                        f"({domain}/{task_id})"
                    )
                    break

        if obs_str and not terminated:
            if "tool_calls" in parsed_message and parsed_message["tool_calls"]:
                obs_content = f"<tool_response>\n{obs_str}\n</tool_response>"
            else:
                obs_content = obs_str
            conversation.append({"role": "user", "content": obs_content})

            new_turn_tokens = renderer.encode_new_turn(obs_content)
            running_tokens.extend(new_turn_tokens)

        if terminated:
            final_reward = reward
            ri = step_info.get("reward_info")
            if ri is not None:
                try:
                    reward_info = json.loads(ri) if isinstance(ri, str) else ri
                except (json.JSONDecodeError, TypeError):
                    reward_info = {"raw": str(ri)}

            structured_messages, structured_steps = _extract_structured_data(
                gym_env, step_info,
            )

            if use_action_reward and reward_info:
                original_reward = final_reward
                final_reward = _recompute_reward_with_action(
                    reward_info, final_reward,
                )
                if final_reward != original_reward:
                    logger.debug(
                        f"ACTION reward: {original_reward:.1f} -> "
                        f"{final_reward:.1f} ({domain}/{task_id})"
                    )
                if transitions:
                    transitions[-1] = Transition(
                        ob=transitions[-1].ob,
                        ac=transitions[-1].ac,
                        reward=final_reward,
                        episode_done=True,
                    )
            else:
                original_reward = final_reward

            # Embed the final step (action that led to termination)
            if embed_client is not None and not early_stopped:
                action_text = _action_text_from_parsed(parsed_message)
                # Terminal step: use "<empty>" for response since there's no next obs
                act_emb, res_emb = await embed_client.embed_step(
                    action_text, "<empty>",
                )
                act_embeddings.append(act_emb)
                res_embeddings.append(res_emb)

            termination_reason = "agent_stop"
            break

    del gym_env

    duration = time.time() - t_start
    # Per-rollout phase composition (Tier 2). Shares are of *measured* phase
    # time so they sum to ~1.0; raw seconds + duration are kept so callers can
    # recompute against wall-clock. score is exact; gen/env/embed are
    # elapsed-with-contention (see accumulator note above).
    _phase_total = (
        gen_elapsed_s + env_elapsed_s + embed_wall_clock_s + score_wall_clock_s
    )
    phase_pct = {
        "gen": gen_elapsed_s / _phase_total if _phase_total > 0 else None,
        "env": env_elapsed_s / _phase_total if _phase_total > 0 else None,
        "embed": embed_wall_clock_s / _phase_total if _phase_total > 0 else None,
        "score": score_wall_clock_s / _phase_total if _phase_total > 0 else None,
    }

    episode_meta = {
        "domain": domain,
        "task_id": str(task_id),
        "seed": seed,
        "reward": final_reward,
        "original_reward": original_reward if use_action_reward else final_reward,
        "reward_info": reward_info,
        "termination_reason": termination_reason,
        "num_turns": len(transitions),
        "prefill_tokens": total_prefill,
        "sample_tokens": total_sample,
        "duration": duration,
        "messages": structured_messages if (terminated and not early_stopped) else conversation,
        "steps": structured_steps if (terminated and not early_stopped) else None,
        "conversation": conversation,
        "early_stopped": early_stopped,
        "early_stop_turn": early_stop_turn,
        "predicted_reward": final_reward if early_stopped else None,
        "prm_scores": prm_scores,
        "step_act_embeddings": act_embeddings,
        "step_res_embeddings": res_embeddings,
        "embed_wall_clock_s": embed_wall_clock_s,
        "score_wall_clock_s": score_wall_clock_s,
        "gen_elapsed_s": gen_elapsed_s,
        "env_elapsed_s": env_elapsed_s,
        "phase_pct": phase_pct,
    }

    return transitions, final_reward, conversation, episode_meta


# ── Group + batch collection with early stopping ─────────────────────


async def collect_group_with_early_stopping(
    domain: str,
    task_id: str,
    task: object,
    group_size: int,
    sampling_client: Any,
    renderer: Renderer,
    max_tokens: int,
    temperature: float,
    max_turns: int,
    user_model: str,
    base_seed: int = 0,
    cost_tracker: CostTracker | None = None,
    use_action_reward: bool = False,
    *,
    prm: Any,
    embed_client: EmbeddingClient,
    support_manager: SupportSetManager,
    es_config: EarlyStopConfig,
    batch_idx: int = 0,
    rollout_sem: asyncio.Semaphore | None = None,
) -> tuple[TrajectoryGroup, list[dict], dict]:
    """Collect a group of rollouts with optional early stopping.

    Returns (group, episode_metas, es_stats) where es_stats summarizes
    early stopping activity for this group. If rollout_sem is given, it
    caps how many rollouts run concurrently across the whole batch.
    """
    t_group_start = time.time()
    instance_id = f"{domain}_{task_id}"
    can_early_stop = support_manager.has_viable_support(instance_id)

    prm_copy = None
    if can_early_stop:
        prm_copy = copy.copy(prm)
        sup_act, sup_res, sup_labels = support_manager.get_support(instance_id)
        prm_copy.set_support(sup_act, sup_res, sup_labels)

    min_completed = es_config.min_completed

    async def _run_rollout(rep: int):
        coro = rollout_episode_with_early_stopping(
            domain=domain,
            task_id=task_id,
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=max_tokens,
            temperature=temperature,
            max_turns=max_turns,
            user_model=user_model,
            seed=base_seed + rep,
            cost_tracker=cost_tracker,
            use_action_reward=use_action_reward,
            prm=prm_copy,
            embed_client=embed_client,
            threshold=es_config.threshold,
            min_steps=es_config.min_steps,
            can_early_stop=can_early_stop and rep >= min_completed,
        )
        if rollout_sem is None:
            return await coro
        async with rollout_sem:
            return await coro

    results = await asyncio.gather(
        *[_run_rollout(rep) for rep in range(group_size)],
        return_exceptions=True,
    )

    trajectories: list[Trajectory] = []
    final_rewards: list[float] = []
    episode_metas: list[dict] = []

    n_early = 0
    n_completed = 0
    turns_at_stop = []
    turns_completed = []

    for rep, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                f"Rollout {instance_id} rep={rep} failed: "
                f"{type(result).__name__}: {result}"
            )
            episode_metas.append({
                "domain": domain,
                "task_id": str(task_id),
                "rep": rep,
                "error": str(result),
            })
            continue
        transitions, reward, conversation, meta = result
        meta["rep"] = rep
        episode_metas.append(meta)
        if not transitions:
            logger.warning(
                f"Rollout {instance_id} rep={rep} produced no transitions"
            )
            continue

        traj = Trajectory(
            trajectory_id=TrajectoryID(
                instance_id=instance_id, repetition_id=rep,
            ),
            transitions=transitions,
            final_ob=conversation,
        )
        trajectories.append(traj)
        final_rewards.append(reward)

        # Add non-early-stopped rollouts to support set (growing)
        if meta.get("early_stopped"):
            n_early += 1
            if meta.get("early_stop_turn") is not None:
                turns_at_stop.append(meta["early_stop_turn"])
        else:
            n_completed += 1
            turns_completed.append(meta.get("num_turns", 0))
            if meta.get("termination_reason") != "agent_stop":
                continue
            act_embs = meta.get("step_act_embeddings", [])
            res_embs = meta.get("step_res_embeddings", [])
            if act_embs and res_embs:
                act_seq = np.stack(act_embs)
                res_seq = np.stack(res_embs)
                label = 1 if reward >= 0.5 else 0
                support_manager.stage_rollout(instance_id, act_seq, res_seq, label, batch_idx=batch_idx)

        meta.pop("step_act_embeddings", None)
        meta.pop("step_res_embeddings", None)
        results[rep] = None

    group = TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=final_rewards,
    )

    group_duration = time.time() - t_group_start
    durations_early = [
        m.get("duration", 0) for m in episode_metas
        if m.get("early_stopped") and "duration" in m
    ]
    durations_completed = [
        m.get("duration", 0) for m in episode_metas
        if not m.get("early_stopped") and not m.get("error") and "duration" in m
    ]

    es_stats = {
        "task": instance_id,
        "had_support": can_early_stop,
        "n_early": n_early,
        "n_completed": n_completed,
        "avg_turn_at_stop": (
            sum(turns_at_stop) / len(turns_at_stop)
            if turns_at_stop else None
        ),
        "avg_turns_completed": (
            sum(turns_completed) / len(turns_completed)
            if turns_completed else None
        ),
        "group_duration": group_duration,
        "avg_duration_early": (
            sum(durations_early) / len(durations_early)
            if durations_early else None
        ),
        "avg_duration_completed": (
            sum(durations_completed) / len(durations_completed)
            if durations_completed else None
        ),
    }

    return group, episode_metas, es_stats


async def collect_batch_with_early_stopping(
    batch_tasks: list[tuple[str, object]],
    group_size: int,
    sampling_client: Any,
    renderer: Renderer,
    max_tokens: int,
    temperature: float,
    max_turns: int,
    user_model: str,
    base_seed: int = 0,
    cost_tracker: CostTracker | None = None,
    rollout_dir: Path | None = None,
    batch_idx: int = 0,
    use_action_reward: bool = False,
    *,
    prm: Any,
    embed_client: EmbeddingClient,
    support_manager: SupportSetManager,
    es_config: EarlyStopConfig,
    max_concurrent_rollouts: int = 64,
) -> tuple[list[tuple[str, TrajectoryGroup]], dict]:
    """Collect rollout groups for a batch with early stopping.

    Returns (keyed_groups, batch_es_stats). max_concurrent_rollouts caps how
    many rollouts run at once across the whole batch.
    """
    rollout_sem = asyncio.Semaphore(max_concurrent_rollouts)

    async def _collect_one(
        i: int, domain: str, task,
    ) -> tuple[str, TrajectoryGroup, list[dict], dict]:
        group, metas, es_stats = await collect_group_with_early_stopping(
            domain=domain,
            task_id=task.id,
            task=task,
            group_size=group_size,
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=max_tokens,
            temperature=temperature,
            max_turns=max_turns,
            user_model=user_model,
            base_seed=base_seed + i * group_size,
            cost_tracker=cost_tracker,
            use_action_reward=use_action_reward,
            prm=prm,
            embed_client=embed_client,
            support_manager=support_manager,
            es_config=es_config,
            batch_idx=batch_idx,
            rollout_sem=rollout_sem,
        )
        return f"{domain}_{task.id}", group, metas, es_stats

    coros = [
        _collect_one(i, domain, task)
        for i, (domain, task) in enumerate(batch_tasks)
    ]

    groups: list[tuple[str, TrajectoryGroup]] = []
    all_metas: list[dict] = []
    all_es_stats: list[dict] = []

    import sys
    from tqdm import tqdm
    rollout_pbar = tqdm(
        total=len(batch_tasks),
        desc="  Rollouts",
        unit="group",
        leave=False,
        file=sys.stdout,
    )
    for fut in asyncio.as_completed(coros):
        task_key, group, metas, es_stats = await fut
        all_metas.extend(metas)
        all_es_stats.append(es_stats)
        if group.trajectories_G:
            groups.append((task_key, group))
        rollout_pbar.update(1)
    rollout_pbar.close()
    del coros

    if rollout_dir is not None:
        rollout_dir.mkdir(parents=True, exist_ok=True)
        out_path = rollout_dir / f"batch_{batch_idx:06d}.jsonl"
        with open(out_path, "w") as f:
            for meta in all_metas:
                serializable = {
                    k: v for k, v in meta.items()
                    if k not in ("step_act_embeddings", "step_res_embeddings")
                }
                f.write(json.dumps(serializable, default=str) + "\n")
        logger.info(f"Saved {len(all_metas)} rollout records to {out_path}")

    # Aggregate per-rollout phase timing across the batch (Tier 2). Averages
    # of per-rollout seconds and per-rollout phase shares — averaging ratios
    # avoids the N x double-count that summing concurrent wall-clocks would
    # cause. Excludes errored rollouts (no duration).
    _timed = [m for m in all_metas if not m.get("error") and "duration" in m]

    def _avg(key: str):
        vals = [m[key] for m in _timed if m.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _avg_pct(sub: str):
        vals = [
            m["phase_pct"][sub]
            for m in _timed
            if m.get("phase_pct") and m["phase_pct"].get(sub) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    rollout_timing = {
        "n_rollouts": len(_timed),
        "avg_duration_s": _avg("duration"),
        "avg_gen_elapsed_s": _avg("gen_elapsed_s"),
        "avg_env_elapsed_s": _avg("env_elapsed_s"),
        "avg_embed_elapsed_s": _avg("embed_wall_clock_s"),
        "avg_score_elapsed_s": _avg("score_wall_clock_s"),
        "avg_pct_gen": _avg_pct("gen"),
        "avg_pct_env": _avg_pct("env"),
        "avg_pct_embed": _avg_pct("embed"),
        "avg_pct_score": _avg_pct("score"),
    }

    all_metas.clear()

    # Aggregate early stopping stats
    tasks_with_support = sum(1 for s in all_es_stats if s["had_support"])
    tasks_new = sum(1 for s in all_es_stats if not s["had_support"])
    total_early = sum(s["n_early"] for s in all_es_stats)
    total_completed = sum(s["n_completed"] for s in all_es_stats)
    all_turns_at_stop = [
        s["avg_turn_at_stop"]
        for s in all_es_stats
        if s["avg_turn_at_stop"] is not None
    ]
    all_turns_completed = [
        s["avg_turns_completed"]
        for s in all_es_stats
        if s["avg_turns_completed"] is not None
    ]

    group_durations = [s["group_duration"] for s in all_es_stats]

    batch_es_stats = {
        "tasks_with_support": tasks_with_support,
        "tasks_new": tasks_new,
        "rollouts_early_stopped": total_early,
        "rollouts_completed": total_completed,
        "early_stop_rate": (
            total_early / max(total_early + total_completed, 1)
        ),
        "avg_turn_at_stop": (
            sum(all_turns_at_stop) / len(all_turns_at_stop)
            if all_turns_at_stop else None
        ),
        "avg_turns_completed": (
            sum(all_turns_completed) / len(all_turns_completed)
            if all_turns_completed else None
        ),
        "embedding_texts": embed_client.total_texts,
        "embedding_cost_usd": embed_client.estimated_cost,
        "support_viable_tasks": support_manager.viable_task_count(),
        "support_total_tasks": support_manager.task_count(),
        "per_group": all_es_stats,
        "avg_group_duration": sum(group_durations) / len(group_durations) if group_durations else None,
        "max_group_duration": max(group_durations) if group_durations else None,
        "rollout_timing": rollout_timing,
    }

    return groups, batch_es_stats
