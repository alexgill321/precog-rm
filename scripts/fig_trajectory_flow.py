#!/usr/bin/env python3
"""Figure 1 (fig:trajectory-flow): trajectory flow through paired
action--observation state clusters for one task.

Loads the task's action/observation embeddings from the released corpus,
clusters all steps of all rollouts with HDBSCAN over cosine of the concatenated
[a_t || o_t] step vector (min_cluster_size=5, unassigned steps -> one outlier
state; Appendix A), optionally truncates every rollout to its first N steps,
and renders the alluvial flow (bar height = cluster occupancy, bar colour =
empirical success rate, ribbons = individual rollouts coloured by outcome).

The paper's figure is tau2/qwen `retail/103`, 100 rollouts, first 30 steps.
The published PDF was produced with an earlier, hand-tuned variant of this
renderer (it additionally annotated mean cumulative generated tokens per step),
so the regenerated figure matches in content but not pixel-for-pixel.

    python scripts/fig_trajectory_flow.py --cell tau2:qwen --task retail/103 --steps 30 --max-rollouts 100
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from _paths import TABLE1, FIGS, require


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="tau2:qwen")
    ap.add_argument("--task", default="retail/103")
    ap.add_argument("--steps", type=int, default=30, help="truncate rollouts to the first N steps")
    ap.add_argument("--max-rollouts", type=int, default=100)
    ap.add_argument("--mcs", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.prm.thought_dag import _discover_tasks, _load_rewards_from_files, render_trajectory_flow
    from src.prm.paired_action_response import hdbscan_cluster_affinity, load_paired_embeddings, paired_similarity

    ds, m = args.cell.split(":")
    data_dir = require(TABLE1 / f"{ds}_{m}", "embedding corpus", "embeddings")
    tasks = {t["task_label"]: t for t in _discover_tasks(str(data_dir), str(data_dir / "embeddings" / "action"))}
    if args.task not in tasks:
        raise SystemExit(f"task {args.task} not found; examples: {sorted(tasks)[:5]}")
    task = tasks[args.task]
    action_npz = task["npz_path"]
    response_npz = str(Path(action_npz).parent.parent / "response" / Path(action_npz).name)
    act, res, indices = load_paired_embeddings(action_npz, response_npz)
    rewards = _load_rewards_from_files(task["tdp_files"])

    # rollout cap (first N rollout ids with a reward) and per-rollout step truncation
    rids = sorted(set(r for r, _ in indices) & set(rewards))[: args.max_rollouts]
    keep = [i for i, (r, s) in enumerate(indices) if r in set(rids) and s < args.steps]
    act, res = act[keep], res[keep]
    indices = [indices[i] for i in keep]
    rewards = {r: rewards[r] for r in rids}

    S, _, _ = paired_similarity(act, res, "mean")
    labels = hdbscan_cluster_affinity(S, min_cluster_size=args.mcs, noise_strategy="outlier_bin")
    K = int(labels.max()) + 1
    sr = float(np.mean([rewards[r] >= 0.5 for r in rids]))
    out = Path(args.out) if args.out else FIGS / f"fig_1_{args.task.replace('/', '_')}_first{args.steps}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = render_trajectory_flow(labels, indices, rewards, output_path=None,
                                 title=f"{ds}/{m} {args.task} (n={len(rids)}, K={K}, sr={sr:.0%}, mcs={args.mcs})",
                                 noise_label=K - 1, return_fig=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"saved {out} ({len(rids)} rollouts, {len(indices)} steps, K={K} states incl. outlier)")


if __name__ == "__main__":
    main()
