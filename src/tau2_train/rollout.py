"""Rollout collection using tau2-bench's AgentGymEnv and Tinker sampling.

System prompt and tool format match the canonical tau2-bench LLMAgent pipeline,
rendered through Qwen3's native chat template for Tinker token generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

import tinker

import tau2_train  # noqa: F401  (ensures tau2-bench is on sys.path)
from tau2.gym import AgentGymEnv, register_gym_agent
from tau2.environment.tool import Tool

from tau2_train.config import CostTracker
from tau2_train.renderer import Renderer, Message
from tau2_train.types import (
    TokensWithLogprobs,
    Trajectory,
    TrajectoryGroup,
    TrajectoryID,
    Transition,
)

logger = logging.getLogger(__name__)

register_gym_agent()

# ---------------------------------------------------------------------------
# Canonical tau2-bench system prompt (from tau2.agent.llm_agent)
# ---------------------------------------------------------------------------

AGENT_INSTRUCTION = (
    "You are a customer service agent that helps the user according to the "
    "<policy> provided below.\n"
    "In each turn you can either:\n"
    "- Send a message to the user.\n"
    "- Make a tool call.\n"
    "You cannot do both at the same time.\n"
    "\n"
    "Try to be helpful and always follow the policy. "
    "Always make sure you generate valid JSON only."
)

SYSTEM_PROMPT_TEMPLATE = (
    "<instructions>\n"
    "{agent_instruction}\n"
    "</instructions>\n"
    "<policy>\n"
    "{domain_policy}\n"
    "</policy>"
)

# ---------------------------------------------------------------------------
# Qwen3-native tool rendering (matches the model's chat template)
# ---------------------------------------------------------------------------

TOOL_PREAMBLE = (
    "# Tools\n"
    "\n"
    "You may call one or more functions to assist with the user query.\n"
    "\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)

TOOL_POSTAMBLE = (
    "</tools>\n"
    "\n"
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


def _build_system_prompt(policy: str, tools: list[Tool]) -> str:
    """Build the system prompt matching canonical tau2-bench + Qwen3 template.

    The system message content uses the exact AGENT_INSTRUCTION and SYSTEM_PROMPT
    template from tau2.agent.llm_agent, followed by the Qwen3 native tool section
    that the model's chat template would normally inject.
    """
    base_content = SYSTEM_PROMPT_TEMPLATE.format(
        agent_instruction=AGENT_INSTRUCTION,
        domain_policy=policy.strip(),
    )
    if not tools:
        return base_content

    tool_lines = []
    for tool in tools:
        tool_lines.append(json.dumps(tool.openai_schema))

    tool_section = (
        TOOL_PREAMBLE
        + "\n"
        + "\n".join(tool_lines)
        + "\n"
        + TOOL_POSTAMBLE
    )
    return base_content + "\n\n" + tool_section


def _format_action_for_gym(parsed_message: Message) -> str:
    """Convert a renderer-parsed Message into a string that AgentGymEnv.step() accepts.

    AgentGymEnv.step() calls parse_action_string() which handles:
    - JSON tool calls: {"name": "...", "arguments": {...}}
    - Plain text content
    """
    if "tool_calls" in parsed_message and parsed_message["tool_calls"]:
        tc = parsed_message["tool_calls"][0]
        return json.dumps({
            "name": tc["name"],
            "arguments": tc.get("arguments", tc.get("args", {})),
        })
    return parsed_message.get("content", "")


def _recompute_reward_with_action(reward_info: dict, original_reward: float) -> float:
    """Recompute reward by multiplying in the ACTION component.

    The gym env evaluates with EvaluationType.ALL but only multiplies
    components in the task's reward_basis (typically DB+COMMUNICATE).
    This function adds ACTION to the effective reward.
    """
    if not reward_info:
        return original_reward

    action_checks = reward_info.get("action_checks")
    if not action_checks:
        return original_reward

    all_matched = all(ac.get("action_match", False) for ac in action_checks)
    action_reward = 1.0 if all_matched else 0.0
    return original_reward * action_reward


def _reconstruct_steps_from_messages(messages: list[dict]) -> list[dict]:
    """Reconstruct TDP-style steps from structured SimulationRun messages.

    Pairs each assistant message with its preceding user/tool input_message
    to produce steps compatible with the embedding pipeline.
    """
    steps = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role in ("user", "tool"):
            input_msg = {"role": role, "content": msg.get("content")}
            i += 1

            if i < len(messages) and messages[i].get("role") == "assistant":
                amsg = messages[i]
                sample = {
                    "content": amsg.get("content"),
                    "tool_calls": amsg.get("tool_calls"),
                    "is_chosen": True,
                    "finish_reason": "tool_calls" if amsg.get("tool_calls") else "stop",
                    "logprobs": None,
                    "token_log_probs": [],
                    "total_log_prob": 0.0,
                }
                steps.append({
                    "step_index": len(steps),
                    "input_message": input_msg,
                    "samples": [sample],
                })
                i += 1
            # tool messages following an assistant tool_call are env responses
            while i < len(messages) and messages[i].get("role") == "tool":
                i += 1
        elif role == "assistant" and not steps:
            sample = {
                "content": msg.get("content"),
                "tool_calls": msg.get("tool_calls"),
                "is_chosen": True,
                "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop",
                "logprobs": None,
                "token_log_probs": [],
                "total_log_prob": 0.0,
            }
            steps.append({
                "step_index": 0,
                "input_message": None,
                "samples": [sample],
            })
            i += 1
        else:
            i += 1

    return steps


def _extract_structured_data(gym_env, step_info: dict) -> tuple[list[dict], list[dict]]:
    """Extract structured messages and TDP steps from the gym env.

    Returns (messages, steps) in the same format as DSV32/consolidated baselines.
    """
    messages = []
    steps = []

    sim_run_json = step_info.get("simulation_run")
    if sim_run_json:
        try:
            sim_data = json.loads(sim_run_json) if isinstance(sim_run_json, str) else sim_run_json
            messages = sim_data.get("messages", [])
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to get TDP steps from the orchestrator
    try:
        orchestrator = gym_env._orchestrator
        if orchestrator:
            state = orchestrator.agent_state
            if state and hasattr(state, "tdp_steps") and state.tdp_steps:
                steps = list(state.tdp_steps)
    except Exception as e:
        logger.debug(f"Could not extract tdp_steps from orchestrator: {e}")

    # Fallback: reconstruct steps from structured messages
    if not steps and messages:
        steps = _reconstruct_steps_from_messages(messages)

    return messages, steps


async def rollout_episode(
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
) -> tuple[list[Transition], float, list[dict], dict]:
    """Run a single rollout episode.

    Returns (transitions, final_reward, conversation, episode_meta) where
    episode_meta contains fields for rollout saving.
    """
    t_start = time.time()
    gym_env = AgentGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_turns,
        user_llm=user_model,
    )

    obs_str, info = await asyncio.to_thread(gym_env.reset, seed)
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

    # Build the initial prompt from the conversation so far (system + first user msg).
    # After this, we carry forward the raw token list and only tokenize NEW observations,
    # avoiding decode→re-encode round-trip misalignment.
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
                f"Context full at turn {turn} ({input_token_count} tokens), ending episode"
            )
            termination_reason = "context_full"
            break
        effective_max_tokens = min(max_tokens, available)

        sampling_params = tinker.SamplingParams(
            stop=renderer.get_stop_sequences(),
            max_tokens=effective_max_tokens,
            temperature=temperature,
        )
        sample_result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )

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

        obs_str, reward, terminated, truncated, step_info = await asyncio.to_thread(
            gym_env.step, action_str,
        )

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

        # Extend running_tokens with the raw sampled tokens so the next turn's
        # prompt is built from exact token IDs, never from re-encoded text.
        running_tokens.extend(response_tokens)

        # If the response was truncated (no stop token), inject <|im_end|> so
        # the chat template stays well-formed for the next turn.
        stop_token = renderer.get_stop_sequences()[0]
        if not response_tokens or response_tokens[-1] != stop_token:
            running_tokens.append(stop_token)

        if obs_str and not terminated:
            if "tool_calls" in parsed_message and parsed_message["tool_calls"]:
                obs_content = f"<tool_response>\n{obs_str}\n</tool_response>"
            else:
                obs_content = obs_str
            conversation.append({"role": "user", "content": obs_content})

            # Tokenize only the NEW observation + next assistant prefill.
            # This produces: \n<|im_start|>user\n{obs}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
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
                final_reward = _recompute_reward_with_action(reward_info, final_reward)
                if final_reward != original_reward:
                    logger.debug(
                        f"ACTION reward: {original_reward:.1f} -> {final_reward:.1f} "
                        f"({domain}/{task_id})"
                    )
                # Update the transition reward to match
                if transitions:
                    transitions[-1] = Transition(
                        ob=transitions[-1].ob,
                        ac=transitions[-1].ac,
                        reward=final_reward,
                        episode_done=True,
                    )
            else:
                original_reward = final_reward

            termination_reason = "agent_stop"
            break

    del gym_env

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
        "duration": time.time() - t_start,
        "messages": structured_messages if terminated else conversation,
        "steps": structured_steps if terminated else None,
        "conversation": conversation,
    }

    return transitions, final_reward, conversation, episode_meta


async def collect_group(
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
    rollout_sem: asyncio.Semaphore | None = None,
) -> tuple[TrajectoryGroup, list[dict]]:
    """Collect a group of rollouts for a single task (one GRPO group).

    Returns (group, episode_metas) where episode_metas is a list of dicts
    for each successful rollout. If rollout_sem is given, it caps how many
    rollouts run concurrently across the whole batch.
    """
    instance_id = f"{domain}_{task_id}"

    async def _run_rollout(rep: int):
        coro = rollout_episode(
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

    for rep, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Rollout {instance_id} rep={rep} failed: {result}")
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
            logger.warning(f"Rollout {instance_id} rep={rep} produced no transitions")
            continue
        traj = Trajectory(
            trajectory_id=TrajectoryID(instance_id=instance_id, repetition_id=rep),
            transitions=transitions,
            final_ob=conversation,
        )
        trajectories.append(traj)
        final_rewards.append(reward)

    group = TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=final_rewards,
    )
    return group, episode_metas


async def collect_batch(
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
    max_concurrent_rollouts: int = 64,
) -> list[tuple[str, TrajectoryGroup]]:
    """Collect rollout groups for a batch of tasks.

    Returns list of (task_key, TrajectoryGroup) tuples.
    If rollout_dir is provided, saves episode metadata as JSONL.
    max_concurrent_rollouts caps how many rollouts run at once across the
    whole batch, bounding peak memory and concurrent API connections.
    """
    rollout_sem = asyncio.Semaphore(max_concurrent_rollouts)

    async def _collect_one(i: int, domain: str, task) -> tuple[str, TrajectoryGroup, list[dict]]:
        group, metas = await collect_group(
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
            rollout_sem=rollout_sem,
        )
        return f"{domain}_{task.id}", group, metas

    coros = [
        _collect_one(i, domain, task)
        for i, (domain, task) in enumerate(batch_tasks)
    ]

    groups: list[tuple[str, TrajectoryGroup]] = []
    all_metas: list[dict] = []
    import sys
    rollout_pbar = tqdm(
        total=len(batch_tasks),
        desc="  Rollouts",
        unit="group",
        leave=False,
        file=sys.stdout,
    )
    for fut in asyncio.as_completed(coros):
        task_key, group, metas = await fut
        all_metas.extend(metas)
        if group.trajectories_G:
            groups.append((task_key, group))
        rollout_pbar.update(1)
    rollout_pbar.close()

    if rollout_dir is not None:
        rollout_dir.mkdir(parents=True, exist_ok=True)
        out_path = rollout_dir / f"batch_{batch_idx:06d}.jsonl"
        with open(out_path, "w") as f:
            for meta in all_metas:
                f.write(json.dumps(meta, default=str) + "\n")
        logger.info(f"Saved {len(all_metas)} rollout records to {out_path}")

    return groups


def upload_to_gcs(
    local_path: Path,
    *,
    bucket_name: str,
    gcs_key: str,
    project: str | None = None,
    content_type: str = "application/jsonl",
) -> str | None:
    """Upload a local file to GCS. Returns gs:// URI or None on failure."""
    try:
        from google.cloud import storage
    except ImportError:
        logger.warning("[gcs] google-cloud-storage not installed, skipping upload")
        return None
    try:
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(local_path), content_type=content_type, timeout=120)
        uri = f"gs://{bucket_name}/{gcs_key}"
        logger.info(f"[gcs] uploaded {local_path.name} -> {uri}")
        return uri
    except Exception as e:
        logger.warning(f"[gcs] upload failed for {local_path.name}: {e}")
        return None
