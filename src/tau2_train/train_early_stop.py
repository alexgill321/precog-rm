"""GRPO training loop with ProtoNetPRM early stopping.

Mirrors train.py but integrates prefix-based early stopping into rollout
collection. Requires an early_stop section in the config YAML.

Usage:
    python -m tau2_train.train_early_stop --config tau2_train/configs/early_stop_retail.yaml
"""

from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["LITELLM_LOG"] = "ERROR"

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
import random
import time
from datetime import datetime
from pathlib import Path

import tinker
from tqdm import tqdm

from tau2_train.config import CostTracker, EarlyStopConfig, TrainConfig
from tau2_train.early_stopping import (
    EmbeddingClient,
    SupportSetManager,
    collect_batch_with_early_stopping,
)
from tau2_train.grpo import (
    assemble_training_data,
    compute_advantages,
    remove_constant_reward_groups,
    save_checkpoint,
    train_step,
)
from tau2_train.renderer import get_renderer
from tau2_train.task_loader import load_tasks, sample_batch
from tau2_train.train import (
    _compute_mean_reward,
    _filter_by_context_length,
    _run_eval,
    _setup_logging,
)

logger = logging.getLogger(__name__)


def _batch_pass_rate(metrics: dict) -> float:
    rewards = metrics.get("per_group_rewards", {})
    total = sum(len(r) for r in rewards.values())
    passes = sum(sum(1 for x in r if x >= 0.5) for r in rewards.values())
    return passes / total * 100 if total else 0


def _log_metrics(metrics_file: Path, metrics: dict, t_start: float) -> None:
    metrics["time_total"] = time.time() - t_start
    with open(metrics_file, "a") as f:
        f.write(json.dumps(metrics, default=str) + "\n")


async def train_with_early_stopping(
    cfg: TrainConfig,
    resume_from: str | None = None,
    start_batch: int = 0,
) -> None:
    _setup_logging(cfg.log_path)

    try:
        import litellm
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        from loguru import logger as loguru_logger
        loguru_logger.disable("litellm")
    except ImportError:
        pass
    logger.info(f"Training config: {cfg}")

    es_cfg = cfg.early_stop
    if es_cfg is None or not es_cfg.checkpoint_path:
        raise ValueError("early_stop config with checkpoint_path is required")

    run_name = cfg.run_name or f"tau2_es_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics_file = Path(cfg.log_path) / "metrics.jsonl"

    # --- Load ProtoNetPRM (forced CPU to avoid MPS memory leak) ---
    from src.prm.cluster_sequence import ProtoNetPRM
    import torch as _torch
    prm = ProtoNetPRM.load(es_cfg.checkpoint_path)
    if hasattr(prm, '_device') and prm._device != _torch.device("cpu"):
        for attr in dir(prm):
            val = getattr(prm, attr, None)
            if isinstance(val, (_torch.nn.Module, _torch.nn.Parameter)):
                val.cpu()
        prm._device = _torch.device("cpu")
    logger.info(f"Loaded ProtoNetPRM from {es_cfg.checkpoint_path} (device={prm._device})")

    # --- Embedding client + support manager ---
    embed_client = EmbeddingClient(model=es_cfg.embedding_model)
    support_manager = SupportSetManager(
        save_dir=es_cfg.support_save_dir,
        rolling_window=es_cfg.rolling_window,
        min_class_support=es_cfg.min_class_support,
    )

    if es_cfg.support_save_dir and Path(es_cfg.support_save_dir).exists():
        support_manager.load(es_cfg.support_save_dir)
        logger.info(
            f"Resumed support sets: {support_manager.task_count()} tasks "
            f"({support_manager.viable_task_count()} viable)"
        )

    if es_cfg.permanent_support_dir:
        perm_dir = Path(es_cfg.permanent_support_dir)
        support_manager.load_permanent_support(
            manifest_path=perm_dir / "manifest.json",
            embeddings_dir=perm_dir / "data" / "embeddings",
            n_per_task=es_cfg.n_permanent_per_task,
        )

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
    session_file.parent.mkdir(parents=True, exist_ok=True)
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
            logger.info(
                f"Loaded {len(eval_tasks)} eval tasks "
                f"(split={cfg.eval_task_split})"
            )
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
        # Tier 1 phase wall-clock. The batch loop is serial (each phase is
        # awaited to completion before the next), so these are non-overlapping
        # and sum to ~time_total. Same dict object stored in metrics, so phases
        # added after early `continue`s still get recorded.
        timings: dict = {}
        metrics["timings"] = timings

        # 1. Sample tasks and collect rollouts with early stopping
        batch_tasks = sample_batch(tasks, cfg.batch_size, rng)
        task_ids_str = [f"{d}_{t.id}" for d, t in batch_tasks]
        metrics["task_ids"] = task_ids_str
        logger.info(
            f"Batch {i_batch}: collecting {cfg.batch_size} tasks x "
            f"{cfg.group_size} rollouts | tasks={task_ids_str}"
        )

        _t_rollout = time.perf_counter()
        keyed_groups, batch_es_stats = await collect_batch_with_early_stopping(
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
            prm=prm,
            embed_client=embed_client,
            support_manager=support_manager,
            es_config=es_cfg,
            max_concurrent_rollouts=cfg.max_concurrent_rollouts,
        )

        # Commit staged support rollouts now that the batch completed
        n_staged = support_manager.commit_staged()
        timings["rollout"] = time.perf_counter() - _t_rollout
        mem.checkpoint("after_rollouts")

        metrics["early_stop"] = batch_es_stats
        logger.info(
            f"  Early stopping: "
            f"{batch_es_stats['rollouts_early_stopped']} early / "
            f"{batch_es_stats['rollouts_completed']} completed | "
            f"support={batch_es_stats['support_viable_tasks']}/"
            f"{batch_es_stats['support_total_tasks']} tasks | "
            f"staged={n_staged}"
        )

        group_keys = [k for k, _ in keyed_groups]
        groups = [g for _, g in keyed_groups]

        metrics["num_groups_before_filter"] = len(groups)
        metrics["avg_reward_before_filter"] = _compute_mean_reward(groups)

        for g_idx, g in enumerate(groups):
            g_task = group_keys[g_idx]
            logger.info(
                f"  Group {g_idx} ({g_task}): rewards={g.final_rewards_G}"
            )
        metrics["per_group_rewards"] = {
            group_keys[i]: g.final_rewards_G
            for i, g in enumerate(groups)
        }

        # 2. Filter constant-reward groups
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
        metrics["train_tokens"] = train_tokens_this_batch

        # 5. Train step
        train_metrics: dict[str, any] = {}
        _t_train = time.perf_counter()
        training_logprobs = await train_step(
            data_D=data_D,
            training_client=training_client,
            learning_rate=cfg.learning_rate,
            num_substeps=cfg.num_substeps,
            loss_fn=cfg.loss_fn,
            loss_fn_config=cfg.loss_fn_config,
            metrics=train_metrics,
        )
        timings["train"] = time.perf_counter() - _t_train
        if train_metrics:
            metrics["train_metrics"] = train_metrics

        # KL divergence
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

        # Break reference cycles: null out heavy data inside trajectories
        # before deleting the containers. Asyncio coroutine frames can pin
        # Transition objects (which hold full token histories) if we only del
        # the top-level references.
        _t_cleanup = time.perf_counter()
        for _, g in keyed_groups:
            for traj in g.trajectories_G:
                for t in traj.transitions:
                    t.ob = None
                    t.ac = None
                traj.transitions.clear()
        del keyed_groups, groups, filtered_groups, data_D, advantages_P, training_logprobs
        del batch_tasks, task_ids_str
        import gc
        gc.collect()
        gc.collect()
        # Clear Tinker SDK lru_caches that accumulate per-request type adapters
        try:
            from tinker._models import _CachedTypeAdapter, strip_annotated_type as _m_sat
            from tinker._utils._transform import _get_annotated_type, get_type_hints, strip_annotated_type as _t_sat
            for cache in [_CachedTypeAdapter, _m_sat, _get_annotated_type, get_type_hints, _t_sat]:
                if hasattr(cache, 'cache_clear'):
                    cache.cache_clear()
        except ImportError:
            pass
        mem.cleanup()
        timings["cleanup"] = time.perf_counter() - _t_cleanup
        mem.checkpoint("after_cleanup")

        support_rollouts = sum(len(v["labels"]) for v in support_manager._supports.values())
        support_bytes = sum(
            a.nbytes + r.nbytes
            for v in support_manager._supports.values()
            for a, r in zip(v["act"], v["res"])
        )
        obj_counts = mem.object_summary()
        mem_dict = mem.to_dict()
        mem_dict["support_rollouts"] = support_rollouts
        mem_dict["support_mb"] = round(support_bytes / 1e6, 1)
        mem_dict["objects"] = obj_counts
        metrics["memory"] = mem_dict
        mem.log_summary(logger)
        logger.info(
            f"  Support: {support_rollouts} rollouts, {support_bytes / 1e6:.0f} MB | "
            f"Objects: ndarray={obj_counts.get('ndarray', 0)} "
            f"Transition={obj_counts.get('Transition', 0)} "
            f"Trajectory={obj_counts.get('Trajectory', 0)}"
        )

        # 6. Checkpoint
        if (i_batch + 1) % cfg.save_every == 0:
            _t_ckpt = time.perf_counter()
            name = f"{run_name}_{i_batch + 1:06d}"
            await save_checkpoint(training_client, name)
            # Record a resume pointer for preemption recovery. Written by the
            # same process that just saved the checkpoint, so the model_id in
            # the path always matches a checkpoint that actually exists —
            # robust across scheduler requeues.
            resume_state_file = Path(cfg.log_path) / "resume_state.txt"
            resume_state_file.write_text(
                f"start_batch={i_batch + 1}\n"
                f"resume_from=tinker://{model_id}/weights/{name}\n"
            )
            # Also save support sets
            if es_cfg.support_save_dir:
                support_manager.save(es_cfg.support_save_dir)
                logger.info(f"  Saved support sets to {es_cfg.support_save_dir}")
            timings["checkpoint"] = time.perf_counter() - _t_ckpt

        # 7. Update sampling client
        _t_sync = time.perf_counter()
        sampling_client._client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )
        timings["weight_sync"] = time.perf_counter() - _t_sync

        # 7b. Eval on held-out split
        if eval_tasks and cfg.eval_every and (i_batch + 1) % cfg.eval_every == 0:
            _t_eval = time.perf_counter()
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
            timings["eval"] = time.perf_counter() - _t_eval
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
        # Residual = time_total minus the bracketed serial phases (cheap glue:
        # setup, filtering, advantages, KL, GCS upload). Computed BEFORE
        # _log_metrics so it lands in the persisted row.
        timings["other"] = max(
            0.0, (time.time() - t_start) - sum(timings.values())
        )
        _log_metrics(metrics_file, metrics, t_start)
        logger.info(
            f"Batch {i_batch} done in {time.time() - t_start:.1f}s | "
            f"reward={metrics.get('avg_reward_after_filter', 0):.3f} | "
            f"groups={metrics.get('num_groups_after_filter', 0)} | "
            f"datums={metrics.get('num_datums', 0)}"
        )
        logger.info(
            "  Timings(s): "
            + " ".join(f"{k}={v:.1f}" for k, v in timings.items())
        )
        _rt = batch_es_stats.get("rollout_timing", {})
        if _rt.get("avg_pct_gen") is not None:
            logger.info(
                "  Rollout phase shares: "
                f"gen={_rt['avg_pct_gen']:.0%} "
                f"env={_rt['avg_pct_env']:.0%} "
                f"embed={_rt['avg_pct_embed']:.0%} "
                f"score={_rt['avg_pct_score']:.0%} "
                f"(avg {_rt['avg_duration_s']:.1f}s/rollout)"
            )
        logger.info(f"  Cost running total: {cost.summary()}")
        logger.info(
            f"  Embedding cost: ${embed_client.estimated_cost:.4f} "
            f"({embed_client.total_texts} texts)"
        )

        batch_pbar.set_postfix(
            rew=f"{metrics.get('avg_reward_after_filter', 0):.2f}",
            grp=f"{metrics.get('num_groups_after_filter', 0)}/{cfg.batch_size}",
            pr=f"{_batch_pass_rate(metrics):.0f}%",
            es=f"{batch_es_stats['early_stop_rate']:.0%}",
            cost=f"${cost.total_cost:.0f}",
        )

        # 8. Upload to GCS
        if cfg.gcs_bucket:
            from tau2_train.rollout import upload_to_gcs
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

    # Final save of support sets
    if es_cfg.support_save_dir:
        support_manager.save(es_cfg.support_save_dir)
        logger.info(f"Final support set save to {es_cfg.support_save_dir}")

    logger.info(f"Training complete. Final cost: {cost.summary()}")
    logger.info(
        f"Embedding cost: ${embed_client.estimated_cost:.4f} "
        f"({embed_client.total_texts} texts)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Tau2 GRPO training with ProtoNetPRM early stopping",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--num-batches", type=int, default=None,
                        help="Override num_batches from config")
    parser.add_argument("--tinker-api-key", type=str, default=None,
                        help="Tinker API key (or set TINKER_API_KEY env var)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Tinker checkpoint to resume from")
    parser.add_argument("--start-batch", type=int, default=0,
                        help="Batch index to start from")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for task sampling and cross-val split")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        overrides = yaml.safe_load(f)

    cfg = TrainConfig()
    es_dict = overrides.pop("early_stop", None)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    if args.num_batches is not None:
        cfg.num_batches = args.num_batches
    if args.seed is not None:
        cfg.seed = args.seed

    if args.tinker_api_key:
        os.environ["TINKER_API_KEY"] = args.tinker_api_key

    if es_dict:
        es_cfg = EarlyStopConfig()
        for k, v in es_dict.items():
            if hasattr(es_cfg, k):
                setattr(es_cfg, k, v)
        cfg.early_stop = es_cfg

    asyncio.run(
        train_with_early_stopping(
            cfg,
            resume_from=args.resume_from,
            start_batch=args.start_batch,
        )
    )


if __name__ == "__main__":
    main()
