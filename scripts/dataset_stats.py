#!/usr/bin/env python3
"""Dataset statistics quoted in Section 4.1 / Appendix D, and Appendix E's
training-pool table (tab:pool-sizes).

Reads the mmap caches (built by build_slim_caches.py + pool_traj_worker.py
--convert), the fixed task split, and the AgentClinic tool-reliance lists.

    python scripts/dataset_stats.py
"""
import json

import numpy as np

from _paths import BENCH_NAME, MODEL_NAME, DATASETS, MODELS, SPLITS, require, MMAP_DIR
import pool_traj_worker as W


def main():
    require(MMAP_DIR, "mmap caches", "caches")
    W.load_all_cells()
    test = {ds: set(v) for ds, v in json.load(open(SPLITS)).items()}

    print("Held-out test split (tasks per benchmark, shared across policies):")
    for ds in ("tau2", "bfclv3", "agentclinic"):
        print(f"  {BENCH_NAME[ds]:12} {len(test[ds])}")

    print("\nMixed-outcome tasks per (benchmark, policy) cell "
          "(success rate in [0.16, 0.84] over <=50 rollouts):")
    for ds in ("tau2", "bfclv3", "agentclinic"):
        print(f"  {BENCH_NAME[ds]:12} " + "  ".join(
            f"{MODEL_NAME[m]}={len(W.CELLS[(ds, m)])}" for m in ("gpt5mini", "qwen", "deepseek")))

    print("\nAgentClinic tool-reliant scenarios (evaluation subset, of mixed-outcome):")
    for m in ("gpt5mini", "qwen", "deepseek"):
        ids = W.ac_subset_ids("tool", m)
        mixed = {int(t["label"].split("/")[-1]) for t in W.CELLS[("agentclinic", m)]}
        print(f"  {MODEL_NAME[m]:11} {len(ids & mixed)} of {len(mixed)}")

    print("\nTable 1 evaluation sets N (test split, >=5 rollouts per class, "
          "AgentClinic restricted to tool-reliant):")
    for ds in ("tau2", "bfclv3", "agentclinic"):
        print(f"  {BENCH_NAME[ds]:12} " + "  ".join(
            f"{MODEL_NAME[m]}={len(W.eval_tasks_for(ds, m, test, 4))}"
            for m in ("deepseek", "qwen", "gpt5mini")))

    print("\nTraining pools (tab:pool-sizes): train-split tasks with >=8 rollouts per class:")
    print(f"  {'':12} " + " ".join(f"{MODEL_NAME[m]:>11}" for m in ("deepseek", "qwen", "gpt5mini")))
    for ds in ("tau2", "bfclv3", "agentclinic"):
        print(f"  {BENCH_NAME[ds]:12} " + " ".join(
            f"{len(W.train_tasks_for(ds, m, test)):>11}" for m in ("deepseek", "qwen", "gpt5mini")))


if __name__ == "__main__":
    main()
