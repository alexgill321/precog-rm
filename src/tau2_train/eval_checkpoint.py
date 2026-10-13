"""Eval-only script: load a Tinker checkpoint and run eval on test split.

Usage:
    python -m tau2_train.eval_checkpoint \
        --checkpoint retail_only_train_split_000005 \
        --eval-task-split test_30 \
        --domains retail \
        --group-size 3
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
import time
from pathlib import Path

import tinker

from tau2_train.config import CostTracker
from tau2_train.renderer import get_renderer
from tau2_train.task_loader import load_tasks
from tau2_train.train import _run_eval, _setup_logging

logger = logging.getLogger(__name__)


async def run_eval_from_checkpoint(
    checkpoint: str,
    model_name: str = "Qwen/Qwen3-8B",
    lora_rank: int = 32,
    renderer_name: str = "qwen3",
    domains: list[str] | None = None,
    eval_task_split: str = "test_30",
    group_size: int = 3,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    max_turns: int = 100,
    user_model: str = "gpt-4o-mini",
    use_action_reward: bool = False,
    output_path: str | None = None,
):
    _setup_logging("logs/_eval_checkpoint")

    try:
        import litellm
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        from loguru import logger as loguru_logger
        loguru_logger.disable("litellm")
    except ImportError:
        pass

    domains = domains or ["retail"]

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        model_name, rank=lora_rank,
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer)

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        load_future = await training_client.load_state_async(checkpoint)
        await load_future
        print("Checkpoint loaded.")
    else:
        print("No checkpoint specified — evaluating base model.")

    sampling_client = await training_client.save_weights_and_get_sampling_client_async()

    eval_tasks = load_tasks(domains, task_split=eval_task_split)
    print(f"Loaded {len(eval_tasks)} eval tasks (split={eval_task_split})")

    cost = CostTracker()

    print(f"Running eval (use_action_reward={use_action_reward}, group_size={group_size})...")
    eval_metrics = await _run_eval(
        eval_tasks=eval_tasks,
        group_size=group_size,
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=max_tokens,
        temperature=temperature,
        max_turns=max_turns,
        user_model=user_model,
        cost_tracker=cost,
        use_action_reward=use_action_reward,
    )

    print(f"\nResults ({checkpoint}, action_reward={use_action_reward}):")
    print(f"  Pass rate: {eval_metrics['pass_rate']:.1f}%")
    print(f"  Avg reward: {eval_metrics['avg_reward']:.3f}")
    print(f"  Tasks: {eval_metrics['num_tasks']}, Samples: {eval_metrics['num_samples']}")
    print(f"  Duration: {eval_metrics['duration']:.0f}s")
    print(f"  Cost: {cost.summary()}")

    if output_path:
        result = {
            "checkpoint": checkpoint,
            "use_action_reward": use_action_reward,
            "eval_task_split": eval_task_split,
            "group_size": group_size,
            "metrics": eval_metrics,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Eval a Tinker checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Tinker checkpoint name (omit for base model)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--domains", nargs="+", default=["retail"])
    parser.add_argument("--eval-task-split", default="test_30")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--user-model", default="gpt-4o-mini")
    parser.add_argument("--use-action-reward", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--tinker-api-key", type=str, default=None)
    args = parser.parse_args()

    if args.tinker_api_key:
        os.environ["TINKER_API_KEY"] = args.tinker_api_key

    asyncio.run(run_eval_from_checkpoint(
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        domains=args.domains,
        eval_task_split=args.eval_task_split,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_turns=args.max_turns,
        user_model=args.user_model,
        use_action_reward=args.use_action_reward,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
