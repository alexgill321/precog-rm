"""CLI runner for T2-Bench TDP sampling.

Loads T2-Bench tasks, runs Z rollouts per task using the TDPAgent
(N samples per agent turn), evaluates via T2-Bench's evaluator,
and saves TDP JSON files.

Usage (Vertex MaaS model):
    python -m src.tau2.runner \
        --domain airline \
        --model qwen3-235b \
        --user-model gpt-4o-mini \
        --Z 5 --N 10 \
        --output-dir exp/tau2/exp_001

Usage (OpenAI-compatible / Anyscale vLLM):
    export OPENAI_COMPAT_BASE_URL=https://<service>.anyscaleuserdata.com/v1
    export OPENAI_COMPAT_API_KEY=...
    python -m src.tau2.runner \
        --domain airline \
        --model allenai/Olmo-3.1-32B-Instruct-DPO \
        --user-model gpt-4o-mini \
        --Z 5 --N 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.run import get_tasks
from .config import Tau2TDPConfig
from .data_types import Tau2TDP
from .tdp_agent import TDPAgent

logger = logging.getLogger(__name__)

_loguru_lock = threading.Lock()
_loguru_initialized = False


def _setup_rollout_log(log_dir: str, task_id: str, tdp_index: int) -> tuple:
    """Route all logging to a per-rollout file, return state for cleanup."""
    global _loguru_initialized
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{task_id}_tdp_{tdp_index:02d}.log")

    file_handler = logging.FileHandler(path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    loguru_sink_id = None
    try:
        from loguru import logger as loguru_logger
        with _loguru_lock:
            if not _loguru_initialized:
                loguru_logger.remove()
                _loguru_initialized = True
            loguru_sink_id = loguru_logger.add(
                path, mode="a", level="DEBUG",
                format="{time} {level} {name}: {message}",
            )
    except ImportError:
        pass

    for name in ("openai", "httpx", "httpcore", "urllib3", "google"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return file_handler, loguru_sink_id


def _teardown_rollout_log(state: tuple):
    file_handler, loguru_sink_id = state
    file_handler.close()
    logging.getLogger().removeHandler(file_handler)

    if loguru_sink_id is not None:
        try:
            from loguru import logger as loguru_logger
            with _loguru_lock:
                loguru_logger.remove(loguru_sink_id)
        except (ImportError, ValueError):
            pass


def _serialize_messages(messages: list) -> list[dict]:
    """Convert T2-Bench Message objects to plain dicts for JSON output."""
    result = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            result.append(msg.model_dump())
        else:
            result.append({"content": str(msg)})
    return result


def run_single_tdp(
    domain: str,
    task: Task,
    tdp_index: int,
    config: Tau2TDPConfig,
    seed: int,
    cost_tracker=None,
    user_cost_tracker=None,
    logprobs: bool = False,
) -> Tau2TDP:
    """Run one TDP rollout: a full T2-Bench simulation with the TDPAgent.

    Uses a per-rollout CostTracker so that ``cost_usd`` in the returned TDP
    reflects only *this* rollout's cost, even under high concurrency.  The
    caller is responsible for merging the per-rollout cost into the shared
    tracker (see ``_run_one``).
    """

    from .vertex_maas import CostTracker as _CT

    local_agent_ct = _CT()
    local_user_ct = _CT()

    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor()

    agent = TDPAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        model=config.model,
        n_samples=config.N,
        temperature=config.temperature,
        selection_strategy=config.selection_strategy,
        cost_tracker=local_agent_ct,
        logprobs=logprobs,
        api_base=config.api_base,
        api_key=config.api_key,
    )

    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None

    UserConstructor = registry.get_user_constructor("user_simulator")
    user = UserConstructor(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=config.user_model,
        llm_args={},
    )

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        seed=seed,
    )

    simulation: SimulationRun = orchestrator.run()

    reward_info = evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
        solo_mode=False,
        domain=domain,
    )
    simulation.reward_info = reward_info
    reward = reward_info.reward if reward_info else None

    reward_info_dict = None
    if reward_info:
        try:
            ri = reward_info.model_dump(mode="json")
            reward_info_dict = ri
        except Exception:
            reward_info_dict = {"reward": reward}

    tdp_steps = agent_state_tdp_steps(orchestrator)

    for msg in simulation.messages:
        if hasattr(msg, 'role') and msg.role == 'user' and hasattr(msg, 'usage') and msg.usage:
            local_user_ct.record(
                config.user_model,
                msg.usage.get("prompt_tokens", 0),
                msg.usage.get("completion_tokens", 0),
            )

    rollout_cost = local_agent_ct.total_cost + local_user_ct.total_cost

    tdp = Tau2TDP(
        task_id=task.id,
        model_id=config.model,
        tdp_index=tdp_index,
        domain=domain,
        steps=tdp_steps,
        messages=_serialize_messages(simulation.messages),
        reward=reward,
        reward_info=reward_info_dict,
        termination_reason=simulation.termination_reason
        if isinstance(simulation.termination_reason, str)
        else simulation.termination_reason.value,
        seed=seed,
        cost_usd=rollout_cost,
    )
    tdp._local_agent_ct = local_agent_ct
    tdp._local_user_ct = local_user_ct
    return tdp


def agent_state_tdp_steps(orchestrator: Orchestrator) -> list:
    """Extract TDP steps from the orchestrator's agent state."""
    state = orchestrator.agent_state
    if state and hasattr(state, "tdp_steps"):
        return list(state.tdp_steps)
    return []


ALL_DOMAINS = ["airline", "retail", "telecom"]


def main():
    parser = argparse.ArgumentParser(
        description="Run T2-Bench TDP sampling (N samples per step, Z rollouts)"
    )
    parser.add_argument("--domain", type=str, default="airline",
                        choices=["airline", "retail", "telecom", "all"])
    parser.add_argument("--model", type=str, default="qwen3-235b",
                        help="Model for agent. Vertex MaaS key (e.g. qwen3-235b) or "
                             "HF model ID for OpenAI-compat endpoints (e.g. allenai/Olmo-3.1-32B-Instruct-DPO)")
    parser.add_argument("--api-base", type=str, default=None,
                        help="OpenAI-compatible base URL for non-Vertex models "
                             "(default: OPENAI_COMPAT_BASE_URL env var)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for --api-base (default: OPENAI_COMPAT_API_KEY env var)")
    parser.add_argument("--user-model", type=str, default="gpt-4o-mini",
                        help="LiteLLM model for user simulator")
    parser.add_argument("--Z", type=int, default=10,
                        help="Number of rollouts per task")
    parser.add_argument("--N", type=int, default=1,
                        help="Number of samples per agent turn")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-split", type=str, default="base")
    parser.add_argument("--task-ids", nargs="+", default=None,
                        help="Run only these task IDs")
    parser.add_argument("--task-file", type=str, default=None,
                        help="JSON file with task IDs (list or {all_tasks: [...]})."
                             " Merged with --task-ids if both given.")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Limit to first N tasks")
    parser.add_argument("--selection-strategy", type=str, default="random",
                        choices=["random", "max_logprob"])
    parser.add_argument("--max-concurrency", type=int, default=1,
                        help="Parallel tasks (each task is sequential internally)")
    parser.add_argument("--logprobs", action="store_true",
                        help="Enable logprobs collection (slower, can cause Vertex hangs)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries per rollout on transient failures")
    parser.add_argument("--output-dir", type=str, default="rollouts/tau2",
                        help="Output directory for TDP JSONs")
    args = parser.parse_args()

    if args.task_file:
        with open(args.task_file) as _tf:
            _td = json.load(_tf)
        file_ids = _td if isinstance(_td, list) else _td.get("all_tasks", _td.get("task_ids", []))
        if args.task_ids:
            file_ids = list(set(file_ids) | set(args.task_ids))
        args.task_ids = file_ids

    # Console: only warnings+. All debug/info goes to per-rollout log files.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for _name in ("openai", "httpx", "httpcore", "urllib3", "google"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    try:
        from loguru import logger as _loguru
        _loguru.remove()
        _loguru.add(lambda msg: None, level="WARNING")
    except ImportError:
        pass

    from .vertex_maas import CostTracker

    cost_tracker = CostTracker()
    user_cost_tracker = CostTracker()

    domains = ALL_DOMAINS if args.domain == "all" else [args.domain]

    config = Tau2TDPConfig(
        Z=args.Z,
        N=args.N,
        temperature=args.temperature,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        user_model=args.user_model,
        domain=args.domain,
        task_split=args.task_split,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        selection_strategy=args.selection_strategy,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        output_dir=args.output_dir,
    )
    config.resolve_paths()

    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    # --- Build flat list of (domain, task, rollout_idx) work items ---
    work_items: list[tuple[str, Task, int]] = []
    for domain in domains:
        domain_task_ids = config.task_ids
        if domain_task_ids:
            # Strip domain prefix (e.g. "airline/5" -> "5") for this domain,
            # and drop IDs belonging to other domains.
            stripped = []
            for tid in domain_task_ids:
                if tid.startswith(f"{domain}/"):
                    stripped.append(tid[len(domain) + 1:])
                elif "/" not in tid:
                    stripped.append(tid)
            domain_task_ids = stripped or None
        tasks = get_tasks(
            task_set_name=domain,
            task_split_name=config.task_split,
            task_ids=domain_task_ids,
            num_tasks=config.num_tasks,
        )
        for task in tasks:
            for z in range(config.Z):
                work_items.append((domain, task, z))

    total_rollouts = len(work_items)
    rng = random.Random(config.seed)
    seeds = [rng.randint(0, 1_000_000) for _ in range(config.Z)]

    log_dir = os.path.join(config.output_dir, "log")
    all_tdps: list[Tau2TDP] = []
    completed = 0
    skipped = 0

    # --- Main progress bar ---
    pbar = tqdm(total=total_rollouts, desc="Overall", unit="rollout", position=0)
    pbar.set_postfix(cost="$0.0000", reward="—")
    progress_lock = threading.Lock()

    # Skip cached rollouts first
    pending_items = []
    for domain, task, z in work_items:
        save_dir = os.path.join(config.output_dir, domain)
        output_path = os.path.join(save_dir, f"{task.id}_tdp_{z:02d}.json")
        if os.path.exists(output_path):
            with open(output_path) as f:
                all_tdps.append(Tau2TDP.from_dict(json.load(f)))
            skipped += 1
            completed += 1
            pbar.update(1)
        else:
            pending_items.append((domain, task, z))

    max_retries = args.max_retries

    def _run_one(domain, task, z):
        save_dir = os.path.join(config.output_dir, domain)
        output_path = os.path.join(save_dir, f"{task.id}_tdp_{z:02d}.json")

        last_exc = None
        for attempt in range(1, max_retries + 1):
            handler = _setup_rollout_log(
                os.path.join(log_dir, domain), task.id, z,
            )
            t0 = time.time()
            try:
                tdp = run_single_tdp(
                    domain=domain,
                    task=task,
                    tdp_index=z,
                    config=config,
                    seed=seeds[z],
                    logprobs=args.logprobs,
                )
            except Exception as exc:
                _teardown_rollout_log(handler)
                last_exc = exc
                if attempt < max_retries:
                    wait = min(2 ** attempt, 60)
                    logger.warning(
                        f"Rollout {domain}/{task.id} z={z} attempt {attempt}/{max_retries} "
                        f"failed ({exc}), retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue
                raise
            else:
                _teardown_rollout_log(handler)
                break

        # Merge per-rollout costs into the shared trackers
        # (CostTracker.record has its own internal lock)
        if hasattr(tdp, '_local_agent_ct'):
            for model, t in tdp._local_agent_ct.totals.items():
                cost_tracker.record(model, t.get("input_tokens", 0),
                                    t.get("output_tokens", 0))
            del tdp._local_agent_ct
        if hasattr(tdp, '_local_user_ct'):
            for model, t in tdp._local_user_ct.totals.items():
                user_cost_tracker.record(model, t.get("input_tokens", 0),
                                          t.get("output_tokens", 0))
            del tdp._local_user_ct

        elapsed = time.time() - t0
        os.makedirs(save_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tdp.to_dict(), f, indent=2, default=str)
        return tdp, elapsed

    failed_rollouts: list[tuple[str, str, int]] = []
    max_workers = config.max_concurrency
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, domain, task, z): (domain, task, z)
            for domain, task, z in pending_items
        }
        for future in as_completed(futures):
            domain, task, z = futures[future]
            try:
                tdp, elapsed = future.result()
            except Exception:
                logger.exception(
                    f"Rollout failed after {max_retries} attempts: {domain}/{task.id} z={z}"
                )
                failed_rollouts.append((domain, task.id, z))
                pbar.update(1)
                continue

            with progress_lock:
                completed += 1
                all_tdps.append(tdp)
                with_reward = [t for t in all_tdps if t.reward is not None]
                avg_reward = (
                    sum(t.reward for t in with_reward) / len(with_reward)
                    if with_reward else 0
                )
                pbar.update(1)
                agent_cost = cost_tracker.total_cost
                user_cost = user_cost_tracker.total_cost
                pbar.set_postfix(
                    agent=f"${agent_cost:.2f}",
                    user=f"${user_cost:.2f}",
                    total=f"${agent_cost + user_cost:.2f}",
                    avg_r=f"{avg_reward:.2f}",
                    last=f"r={tdp.reward} {elapsed:.0f}s",
                )

    pbar.close()

    # --- Final summary ---
    total = len(all_tdps)
    total_tasks = len({(t.domain, t.task_id) for t in all_tdps})
    with_reward = [t for t in all_tdps if t.reward is not None]
    avg_reward = sum(t.reward for t in with_reward) / len(with_reward) if with_reward else 0
    avg_steps = sum(t.trajectory_length for t in all_tdps) / total if total else 0

    combined_cost = cost_tracker.total_cost + user_cost_tracker.total_cost
    print(f"\n{'='*60}")
    print(f"Done: {total} rollouts across {total_tasks} tasks ({', '.join(domains)})")
    print(f"  Avg reward: {avg_reward:.3f}  |  Avg agent turns: {avg_steps:.1f}")
    print(f"  Skipped (cached): {skipped}")
    if failed_rollouts:
        print(f"  FAILED (after {max_retries} retries): {len(failed_rollouts)}")
        for d, tid, z in failed_rollouts[:20]:
            print(f"    {d}/{tid} z={z}")
        if len(failed_rollouts) > 20:
            print(f"    ... and {len(failed_rollouts) - 20} more")
        print("  Re-run the same command to retry failed rollouts.")
    else:
        print(f"  Failed: 0")
    print(f"\nAgent {cost_tracker.summary()}")
    print(f"\nUser {user_cost_tracker.summary()}")
    print(f"\n  Combined total: ${combined_cost:.4f}")
    print(f"{'='*60}")

    summary = {
        "domains": domains,
        "model": config.model,
        "user_model": config.user_model,
        "Z": config.Z,
        "N": config.N,
        "n_tasks": total_tasks,
        "n_tdps": total,
        "avg_reward": avg_reward,
        "avg_agent_turns": avg_steps,
    }
    with open(os.path.join(config.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    cost_data = {
        "agent": {
            "totals": cost_tracker.totals,
            "total_cost_usd": cost_tracker.total_cost,
            "summary": cost_tracker.summary(),
        },
        "user": {
            "totals": user_cost_tracker.totals,
            "total_cost_usd": user_cost_tracker.total_cost,
            "summary": user_cost_tracker.summary(),
        },
        "combined_cost_usd": cost_tracker.total_cost + user_cost_tracker.total_cost,
    }
    with open(os.path.join(config.output_dir, "cost_summary.json"), "w") as f:
        json.dump(cost_data, f, indent=2)


if __name__ == "__main__":
    main()
