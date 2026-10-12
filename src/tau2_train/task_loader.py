"""Load tau2-bench tasks for training."""

from __future__ import annotations

import logging
import random

import tau2_train  # noqa: F401  (ensures tau2-bench is on sys.path)
from tau2.registry import registry

logger = logging.getLogger(__name__)


def load_tasks(
    domains: list[str],
    task_ids: list[str] | None = None,
    task_split: str | None = "base",
) -> list[tuple[str, object]]:
    """Load tau2-bench tasks and return (domain, task) pairs.

    Args:
        domains: Which domains to load (e.g. ["retail"], ["airline", "retail", "telecom"]).
        task_ids: If provided, only include tasks whose id is in this list.
            IDs can be bare (e.g. "5") or domain-prefixed (e.g. "retail_5").
        task_split: Task split name (default "base").

    Returns:
        List of (domain_name, Task) tuples.
    """
    all_tasks: list[tuple[str, object]] = []

    for domain in domains:
        tasks_loader = registry.get_tasks_loader(domain)
        raw_tasks = tasks_loader(task_split)
        logger.info(f"Loaded {len(raw_tasks)} tasks for domain={domain} split={task_split}")

        for task in raw_tasks:
            if task_ids is not None:
                bare_id = str(task.id)
                prefixed_id = f"{domain}_{task.id}"
                if bare_id not in task_ids and prefixed_id not in task_ids:
                    continue
            all_tasks.append((domain, task))

    logger.info(f"Total tasks after filtering: {len(all_tasks)}")
    return all_tasks


def generate_split(
    domain: str,
    seed: int,
    n_test: int = 30,
) -> tuple[list[tuple[str, object]], list[tuple[str, object]]]:
    """Generate a deterministic train/test split of all tasks for a domain.

    Loads all tasks (no split filter), shuffles with the given seed,
    and returns (train_tasks, test_tasks).
    """
    all_tasks = load_tasks([domain], task_split=None)
    rng = random.Random(seed)
    shuffled = list(all_tasks)
    rng.shuffle(shuffled)
    test_tasks = shuffled[:n_test]
    train_tasks = shuffled[n_test:]
    logger.info(
        f"Generated split for {domain} seed={seed}: "
        f"{len(train_tasks)} train, {len(test_tasks)} test"
    )
    return train_tasks, test_tasks


def sample_batch(
    tasks: list[tuple[str, object]],
    batch_size: int,
    rng: random.Random | None = None,
) -> list[tuple[str, object]]:
    """Sample a batch of tasks (with replacement if batch_size > len(tasks))."""
    if rng is None:
        rng = random.Random()
    if batch_size <= len(tasks):
        return rng.sample(tasks, batch_size)
    return rng.choices(tasks, k=batch_size)
