"""Main GRPO training loop for tau2 on Tinker.

Usage:
    python -m flow_eval.tau2_train.train [--config config.yaml]

Or import and call train() directly.
"""

from __future__ import annotations

import os
os.environ.setdefault("LITELLM_LOG", "ERROR")

from pathlib import Path as _Path
_env_file = _Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import argparse
import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import tinker
from tqdm import tqdm

from tau2_train.config import CostTracker, TrainConfig
from tau2_train.grpo import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
    save_checkpoint,
    train_step,
)
from tau2_train.renderer import get_renderer
from tau2_train.rollout import collect_batch, upload_to_gcs
from tau2_train.task_loader import load_tasks, sample_batch
from tau2_train.types import TrajectoryGroup

logger = logging.getLogger(__name__)


class _OurLogsOnly(logging.Filter):
    """Only allow log records from __main__ and tau2_train.*"""
    _ALLOWED = ("__main__", "tau2_train")
    def filter(self, record: logging.LogRecord) -> bool:
        return any(record.name.startswith(p) for p in self._ALLOWED)


def _setup_logging(log_path: str) -> None:
    os.makedirs(log_path, exist_ok=True)

    # File handler: only our training logs (not LiteLLM/hpack noise)
    file_handler = logging.FileHandler(os.path.join(log_path, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(_OurLogsOnly())
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )

    # Console: only warnings from our code (tqdm handles progress)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.addFilter(_OurLogsOnly())
    console_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Suppress all noisy loggers
    for noisy in ["tau2", "litellm", "httpx", "httpcore", "urllib3", "tinker",
                   "gymnasium", "openai", "anthropic", "hpack", "h2"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        from loguru import logger as loguru_logger
        loguru_logger.disable("tau2")
    except ImportError:
        pass


def _compute_mean_reward(groups: list[TrajectoryGroup]) -> float:
    if not groups:
        return 0.0
    total = sum(
        sum(g.get_total_rewards()) / max(len(g.trajectories_G), 1)
        for g in groups
    )
    return total / len(groups)


def _filter_by_context_length(
    groups: list[TrajectoryGroup],
    max_tokens: int,
) -> tuple[list[TrajectoryGroup], int]:
    """Remove trajectories whose observations exceed max_tokens."""
    filtered = []
    removed = 0
    for group in groups:
        kept_trajs = []
        kept_rewards = []
        for i, traj in enumerate(group.trajectories_G):
            exceeds = any(
                hasattr(t.ob, "length") and t.ob.length > max_tokens
                for t in traj.transitions
            )
            if exceeds:
                removed += 1
            else:
                kept_trajs.append(traj)
                kept_rewards.append(group.final_rewards_G[i])
        if kept_trajs:
            filtered.append(TrajectoryGroup(
                trajectories_G=kept_trajs,
                final_rewards_G=kept_rewards,
            ))
    return filtered, removed


async def _run_eval(
    eval_tasks: list[tuple[str, object]],
    group_size: int,
    sampling_client,
    renderer,
    max_tokens: int,
    temperature: float,
    max_turns: int,
    user_model: str,
    cost_tracker: CostTracker | None = None,
    use_action_reward: bool = False,
) -> dict:
    """Run eval on held-out tasks, return summary metrics."""
    from tau2_train.rollout import collect_group

    logger.info(f"Running eval on {len(eval_tasks)} tasks x {group_size} samples")
    t_start = time.time()

    all_rewards: list[float] = []
    per_task: dict[str, float] = {}

    async def _eval_one(domain: str, task) -> tuple[str, list[float]]:
        group, _metas = await collect_group(
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
            base_seed=0,
            cost_tracker=cost_tracker,
            use_action_reward=use_action_reward,
        )
        rewards = group.final_rewards_G if group.trajectories_G else [0.0]
        return f"{domain}_{task.id}", rewards

    coros = [_eval_one(domain, task) for domain, task in eval_tasks]

    import sys
    eval_pbar = tqdm(
        total=len(eval_tasks),
        desc="  Eval",
        unit="task",
        leave=False,
        file=sys.stdout,
    )
    for fut in asyncio.as_completed(coros):
        task_key, task_rewards = await fut
        all_rewards.extend(task_rewards)
        per_task[task_key] = sum(task_rewards) / max(len(task_rewards), 1)
        eval_pbar.update(1)
    eval_pbar.close()

    avg_reward = sum(all_rewards) / max(len(all_rewards), 1)
    pass_rate = sum(1 for r in all_rewards if r >= 0.5) / max(len(all_rewards), 1) * 100

    return {
        "avg_reward": avg_reward,
        "pass_rate": pass_rate,
        "num_tasks": len(eval_tasks),
        "num_samples": len(all_rewards),
        "per_task": per_task,
        "duration": time.time() - t_start,
    }


async def train(
    cfg: TrainConfig,
    resume_from: str | None = None,
    start_batch: int = 0,
) -> None:
    _setup_logging(cfg.log_path)
    logger.info(f"Training config: {cfg}")

    run_name = cfg.run_name or f"tau2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics_file = Path(cfg.log_path) / "metrics.jsonl"

    # --- Tinker setup ---
    service_client = tinker.ServiceClient()
    if resume_from and resume_from.startswith("tinker://"):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        training_client = await service_client.create_training_client_from_state_async(resume_from)
        logger.info("Checkpoint loaded successfully")
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model_name, rank=cfg.lora_rank,
        )
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            load_future = await training_client.load_state_async(resume_from)
            await load_future
            logger.info("Checkpoint loaded successfully")

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(cfg.renderer_name, tokenizer)
    from tau2_train.refreshable_client import RefreshableSamplingClient
    raw_sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    sampling_client = RefreshableSamplingClient(training_client, raw_sampling_client)
    model_id = training_client.model_id
    logger.info(f"Tinker clients initialized | model_id={model_id}")
    logger.info(f"Resume path: tinker://{model_id}/weights/{run_name}_NNNNNN")

    session_file = Path(cfg.log_path) / "tinker_session.txt"
    session_file.write_text(f"model_id={model_id}\nrun_name={run_name}\n")

    # --- Load tasks ---
    if cfg.task_split and cfg.eval_task_split:
        tasks = load_tasks(cfg.domains, cfg.task_ids, cfg.task_split)
        if not tasks:
            raise ValueError("No tasks loaded. Check domains and task_ids.")
        logger.info(f"Loaded {len(tasks)} tasks")
        eval_tasks = None
        if cfg.eval_every and cfg.eval_task_split != cfg.task_split:
            eval_tasks = load_tasks(cfg.domains, task_split=cfg.eval_task_split)
            logger.info(f"Loaded {len(eval_tasks)} eval tasks (split={cfg.eval_task_split})")
    else:
        from tau2_train.task_loader import generate_split
        tasks, eval_tasks = generate_split(cfg.domains[0], cfg.seed, n_test=30)
        train_ids = [f"{d}_{t.id}" for d, t in tasks]
        test_ids = [f"{d}_{t.id}" for d, t in eval_tasks]
        logger.info(f"Cross-val split (seed={cfg.seed}): train={train_ids}")
        logger.info(f"Cross-val split (seed={cfg.seed}): test={test_ids}")

    rng = random.Random(cfg.task_seed if cfg.task_seed is not None else cfg.seed)
    cost = CostTracker()

    rollout_dir = Path(cfg.log_path) / "rollouts" if cfg.save_rollouts else None

    # Advance RNG past already-completed batches to keep sampling deterministic
    if start_batch > 0:
        logger.info(f"Skipping to batch {start_batch} (advancing RNG)")
        for _ in range(start_batch):
            sample_batch(tasks, cfg.batch_size, rng)

    # --- Training loop ---
    import sys
    batch_pbar = tqdm(
        range(start_batch, cfg.num_batches),
        desc="Batches",
        unit="batch",
        initial=start_batch,
        total=cfg.num_batches,
        file=sys.stdout,
    )
    from tau2_train.memory_tracker import MemoryTracker
    mem = MemoryTracker()

    for i_batch in batch_pbar:
        t_start = time.time()
        mem.reset()
        mem.checkpoint("batch_start")

        # Flush litellm's cached httpx client at the start of every batch.
        # litellm pins one OpenAI client (one TCP connection pool) for
        # _DEFAULT_TTL_FOR_HTTPX_CLIENTS = 3600s. On CHPC the stateful firewall
        # silently ages out long-lived TCP flows ~1h in, black-holing the pool
        # and turning every gpt-4o-mini call into an instant "Connection error."
        # Rebuilding fresh per batch (~16 min) keeps connections well under the
        # firewall idle window.
        import litellm
        litellm.in_memory_llm_clients_cache.flush_cache()

        metrics: dict = {"batch": i_batch, "lr": cfg.learning_rate}

        # 1. Sample tasks and collect rollouts
        batch_tasks = sample_batch(tasks, cfg.batch_size, rng)
        task_ids_str = [f"{d}_{t.id}" for d, t in batch_tasks]
        metrics["task_ids"] = task_ids_str
        logger.info(
            f"Batch {i_batch}: collecting {cfg.batch_size} tasks x "
            f"{cfg.group_size} rollouts | tasks={task_ids_str}"
        )

        keyed_groups = await collect_batch(
            batch_tasks=batch_tasks,
            group_size=cfg.group_size,
            sampling_client=sampling_client,
            renderer=renderer,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            max_turns=cfg.max_turns,
            user_model=cfg.user_model,
            base_seed=i_batch * cfg.batch_size * cfg.group_size,
            cost_tracker=cost,
            rollout_dir=rollout_dir,
            batch_idx=i_batch,
            use_action_reward=cfg.use_action_reward,
            max_concurrent_rollouts=cfg.max_concurrent_rollouts,
        )
        mem.checkpoint("after_rollouts")

        group_keys = [k for k, _ in keyed_groups]
        groups = [g for _, g in keyed_groups]

        metrics["num_groups_before_filter"] = len(groups)
        metrics["avg_reward_before_filter"] = _compute_mean_reward(groups)

        # Log per-group rewards for diagnostics
        for g_idx, g in enumerate(groups):
            g_task = group_keys[g_idx]
            logger.info(
                f"  Group {g_idx} ({g_task}): rewards={g.final_rewards_G}"
            )
        metrics["per_group_rewards"] = {
            group_keys[i]: g.final_rewards_G
            for i, g in enumerate(groups)
        }

        # 2. Filter constant-reward groups (zero gradient signal)
        filtered_groups = remove_constant_reward_groups(groups)
        metrics["num_groups_after_filter"] = len(filtered_groups)
        metrics["avg_reward_after_filter"] = _compute_mean_reward(filtered_groups)

        if not filtered_groups:
            logger.warning(
                f"Batch {i_batch}: all groups filtered. Skipping."
            )
            _log_metrics(metrics_file, metrics, t_start)
            continue

        # 3. Filter by context length
        filtered_groups, ctx_removed = _filter_by_context_length(
            filtered_groups, cfg.max_trajectory_tokens,
        )
        metrics["trajectories_removed_context"] = ctx_removed

        if not filtered_groups:
            logger.warning(
                f"Batch {i_batch}: all trajectories exceeded context length. "
                "Skipping."
            )
            _log_metrics(metrics_file, metrics, t_start)
            continue

        # 4. Compute advantages and assemble training data
        advantages_P = compute_advantages(filtered_groups)
        data_D = assemble_training_data(filtered_groups, advantages_P)
        metrics["num_datums"] = len(data_D)

        train_tokens_this_batch = sum(d.model_input.length for d in data_D)
        cost.train_tokens += train_tokens_this_batch

        # 5. Train step (pipelined fwd-bwd + optim via tinker_cookbook)
        train_metrics: dict[str, any] = {}
        training_logprobs = await train_step(
            data_D=data_D,
            training_client=training_client,
            learning_rate=cfg.learning_rate,
            num_substeps=cfg.num_substeps,
            loss_fn=cfg.loss_fn,
            loss_fn_config=cfg.loss_fn_config,
            metrics=train_metrics,
        )
        if train_metrics:
            metrics["train_metrics"] = train_metrics

        # Compute KL between sampled and trained logprobs as a training signal proxy
        import torch
        kl_values = []
        for datum, train_lp in zip(data_D, training_logprobs):
            sample_lp = datum.loss_fn_inputs["logprobs"].to_torch()
            mask = datum.loss_fn_inputs["mask"].to_torch()
            diff = (sample_lp - train_lp) * mask
            if mask.sum() > 0:
                kl_values.append(diff.sum().item() / mask.sum().item())
        if kl_values:
            avg_kl = sum(kl_values) / len(kl_values)
            metrics["kl_sample_train"] = avg_kl
            logger.info(f"  KL(sample||train): {avg_kl:.4f}")

        mem.checkpoint("after_train")
        del keyed_groups, groups, filtered_groups, data_D, advantages_P, training_logprobs
        mem.cleanup()
        mem.checkpoint("after_cleanup")
        metrics["memory"] = mem.to_dict()
        mem.log_summary(logger)

        # 6. Checkpoint
        if (i_batch + 1) % cfg.save_every == 0:
            name = f"{run_name}_{i_batch + 1:06d}"
            await save_checkpoint(training_client, name)

        # 7. Update sampling client with new weights
        sampling_client._client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )

        # 7b. Eval on held-out split
        if eval_tasks and cfg.eval_every and (i_batch + 1) % cfg.eval_every == 0:
            eval_metrics = await _run_eval(
                eval_tasks=eval_tasks,
                group_size=cfg.eval_group_size,
                sampling_client=sampling_client,
                renderer=renderer,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                max_turns=cfg.max_turns,
                user_model=cfg.user_model,
                cost_tracker=cost,
                use_action_reward=cfg.use_action_reward,
            )
            metrics["eval"] = eval_metrics
            logger.info(
                f"  Eval ({cfg.eval_task_split}): "
                f"reward={eval_metrics['avg_reward']:.3f} | "
                f"pass_rate={eval_metrics['pass_rate']:.1f}% | "
                f"tasks={eval_metrics['num_tasks']}"
            )

        metrics["cost_prefill"] = cost.prefill_cost
        metrics["cost_sample"] = cost.sample_cost
        metrics["cost_train"] = cost.train_cost
        metrics["cost_total"] = cost.total_cost
        _log_metrics(metrics_file, metrics, t_start)
        logger.info(
            f"Batch {i_batch} done in {time.time() - t_start:.1f}s | "
            f"reward={metrics.get('avg_reward_after_filter', 0):.3f} | "
            f"groups={metrics.get('num_groups_after_filter', 0)} | "
            f"datums={metrics.get('num_datums', 0)}"
        )
        logger.info(f"  Cost running total: {cost.summary()}")

        batch_pbar.set_postfix(
            rew=f"{metrics.get('avg_reward_after_filter', 0):.2f}",
            grp=f"{metrics.get('num_groups_after_filter', 0)}/{cfg.batch_size}",
            pr=f"{_batch_pass_rate(metrics):.0f}%",
            cost=f"${cost.total_cost:.0f}",
        )

        # 8. Upload to GCS
        if cfg.gcs_bucket:
            gcs_kw = dict(
                bucket_name=cfg.gcs_bucket,
                project=cfg.gcs_project,
            )
            gcs_run_prefix = f"{cfg.gcs_prefix}/{run_name}"
            if rollout_dir is not None:
                rollout_file = rollout_dir / f"batch_{i_batch:06d}.jsonl"
                if rollout_file.exists():
                    upload_to_gcs(
                        rollout_file,
                        gcs_key=f"{gcs_run_prefix}/rollouts/{rollout_file.name}",
                        **gcs_kw,
                    )
            upload_to_gcs(
                metrics_file,
                gcs_key=f"{gcs_run_prefix}/metrics.jsonl",
                content_type="application/jsonl",
                **gcs_kw,
            )

    logger.info(f"Training complete. Final cost: {cost.summary()}")


def _batch_pass_rate(metrics: dict) -> float:
    rewards = metrics.get("per_group_rewards", {})
    total = sum(len(r) for r in rewards.values())
    passes = sum(sum(1 for x in r if x >= 0.5) for r in rewards.values())
    return passes / total * 100 if total else 0


def _log_metrics(metrics_file: Path, metrics: dict, t_start: float) -> None:
    metrics["time_total"] = time.time() - t_start
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tau2 GRPO training on Tinker")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--base-url", type=str, help="Tinker service URL")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--log-path", type=str, default=None)
    parser.add_argument("--user-model", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Tinker checkpoint name or path to resume from")
    parser.add_argument("--start-batch", type=int, default=0,
                        help="Batch index to start from (skips earlier batches)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for task sampling and cross-val split")
    args = parser.parse_args()

    cfg = TrainConfig()

    if args.config:
        import yaml
        with open(args.config) as f:
            overrides = yaml.safe_load(f)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if args.base_url:
        cfg.base_url = args.base_url
    if args.model_name:
        cfg.model_name = args.model_name
    if args.lora_rank is not None:
        cfg.lora_rank = args.lora_rank
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.group_size is not None:
        cfg.group_size = args.group_size
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_batches is not None:
        cfg.num_batches = args.num_batches
    if args.domains is not None:
        cfg.domains = args.domains
    if args.task_ids is not None:
        cfg.task_ids = args.task_ids
    if args.log_path is not None:
        cfg.log_path = args.log_path
    if args.user_model is not None:
        cfg.user_model = args.user_model
    if args.seed is not None:
        cfg.seed = args.seed

    asyncio.run(train(cfg, resume_from=args.resume_from, start_batch=args.start_batch))


if __name__ == "__main__":
    main()
