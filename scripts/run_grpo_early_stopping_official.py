#!/usr/bin/env python3
"""Simulate prefix early stopping on official GRPO baseline rollouts.

Uses pre-computed embeddings and prefill token counts from embed_baseline_rollouts.py.
Runs rolling support configurations only, with compute savings measured in
cumulative prefill tokens (not step count).

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python src/experiments/run_grpo_early_stopping_official.py \
        --checkpoint $PRECOG_DATA/ckpts/prefix/grpo_controller_model_prefix.pt \
        --seed 1 \
        --thresholds 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import TABLE1, REPLAY, GRPO_REPLAY  # noqa: E402
from src.prm.cluster_sequence import ProtoNetPRM

CON_DATA = TABLE1 / "tau2_qwen"            # released tau2/qwen cell (manifest.json + embeddings/)
CON_EMB = CON_DATA / "embeddings"
OUT_DIR = REPLAY
MIN_CLASS_SUPPORT = 16

DEFAULT_THRESHOLDS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def _load_task_embeddings(stem: str, emb_dir: Path):
    """Load embeddings for one batch-task from NPZ files."""
    act_path = emb_dir / "action" / f"{stem}.npz"
    res_path = emb_dir / "response" / f"{stem}.npz"
    if not act_path.exists() or not res_path.exists():
        return None

    act_npz = np.load(str(act_path))
    res_npz = np.load(str(res_path))

    rollouts = {}
    for key in act_npz.files:
        m = re.match(r"r(\d+)_s(\d+)", key)
        if m:
            rid, sid = int(m.group(1)), int(m.group(2))
            rollouts.setdefault(rid, {"act": {}, "res": {}})
            rollouts[rid]["act"][sid] = act_npz[key].squeeze()
    for key in res_npz.files:
        m = re.match(r"r(\d+)_s(\d+)", key)
        if m:
            rid, sid = int(m.group(1)), int(m.group(2))
            rollouts.setdefault(rid, {"act": {}, "res": {}})
            rollouts[rid]["res"][sid] = res_npz[key].squeeze()

    result = {}
    for rid, data in rollouts.items():
        steps = sorted(set(data["act"].keys()) & set(data["res"].keys()))
        if not steps:
            continue
        act_seq = np.stack([data["act"][s] for s in steps])
        res_seq = np.stack([data["res"][s] for s in steps])
        result[rid] = (act_seq, res_seq)
    return result


def _load_task_meta(stem: str, data_dir: Path):
    """Load rewards and prefill tokens from metadata JSON."""
    meta_path = data_dir / f"{stem}.json"
    if not meta_path.exists():
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    rewards = {int(k): v for k, v in meta.get("rewards", {}).items()}
    prefill = {int(k): v for k, v in meta.get("prefill_tokens", {}).items()}
    return rewards, prefill


def _discover_tasks(data_dir: Path):
    """Discover batch-task structure from metadata files."""
    canon_to_batches = defaultdict(list)
    for meta_file in sorted(data_dir.glob("batch_*_retail_*.json")):
        stem = meta_file.stem
        parts = stem.split("_")
        batch = f"{parts[0]}_{parts[1]}"
        task_id = parts[-1]
        canon = f"retail/{task_id}"
        task_label = f"{batch}/retail/{task_id}"
        canon_to_batches[canon].append((task_label, stem))
    for canon in canon_to_batches:
        canon_to_batches[canon].sort()
    return canon_to_batches


def _load_consolidated_support(canon: str, n_support: int, rng, stratified: bool = False):
    """Load Qwen consolidated rollouts as support (same as original script)."""
    from src.prm.embeddings import task_id_to_filename
    stem = task_id_to_filename(canon)
    act_path = CON_EMB / "action" / f"{stem}.npz"
    res_path = CON_EMB / "response" / f"{stem}.npz"
    if not act_path.exists() or not res_path.exists():
        return None

    manifest_path = CON_DATA / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.load(open(manifest_path))
    entry = manifest.get(canon)
    if entry is None:
        return None

    act_npz = np.load(str(act_path))
    res_npz = np.load(str(res_path))

    rollout_data = []
    for r in entry["rollouts"]:
        rid = r["rid"]
        reward = r.get("reward", 0.0)
        label = 1 if (reward or 0) >= 0.5 else 0

        rid_steps_act = {}
        rid_steps_res = {}
        for key in act_npz.files:
            m = re.match(rf"r{rid}_s(\d+)", key)
            if m:
                rid_steps_act[int(m.group(1))] = act_npz[key].squeeze()
        for key in res_npz.files:
            m = re.match(rf"r{rid}_s(\d+)", key)
            if m:
                rid_steps_res[int(m.group(1))] = res_npz[key].squeeze()

        steps = sorted(set(rid_steps_act.keys()) & set(rid_steps_res.keys()))
        if not steps:
            continue
        act_seq = np.stack([rid_steps_act[s] for s in steps])
        res_seq = np.stack([rid_steps_res[s] for s in steps])
        rollout_data.append((act_seq, res_seq, label))

    if not rollout_data or len(set(rd[2] for rd in rollout_data)) < 2:
        return None

    if n_support < len(rollout_data):
        if stratified:
            pass_idx = [i for i, rd in enumerate(rollout_data) if rd[2] == 1]
            fail_idx = [i for i, rd in enumerate(rollout_data) if rd[2] == 0]
            rng.shuffle(pass_idx)
            rng.shuffle(fail_idx)
            half = n_support // 2
            n_pass = min(half, len(pass_idx))
            n_fail = min(half, len(fail_idx))
            if n_pass < half:
                n_fail = min(n_support - n_pass, len(fail_idx))
            elif n_fail < half:
                n_pass = min(n_support - n_fail, len(pass_idx))
            selected = pass_idx[:n_pass] + fail_idx[:n_fail]
        else:
            perm = rng.permutation(len(rollout_data))
            selected = perm[:n_support]
    else:
        selected = range(len(rollout_data))

    act_list = [rollout_data[i][0] for i in selected]
    res_list = [rollout_data[i][1] for i in selected]
    label_list = [rollout_data[i][2] for i in selected]

    if len(set(label_list)) < 2:
        return None
    return act_list, res_list, label_list


def _early_stop_rollout(prm, act_seq, res_seq, threshold):
    """Score a rollout step by step, return (decision, term_step, total_steps)."""
    T = act_seq.shape[0]
    scores = prm.score_path(path=list(range(T)), act_emb=act_seq, res_emb=res_seq)

    for t, s in enumerate(scores):
        if s > threshold:
            return 1, t, T
        elif (1 - s) > threshold:
            return 0, t, T

    decision = 1 if scores[-1] >= 0.5 else 0
    return decision, T, T


def _prefill_at_step(prefill_tokens: list[int], step: int) -> int:
    """Cumulative prefill tokens up to and including the given step."""
    if not prefill_tokens:
        return 0
    idx = min(step, len(prefill_tokens) - 1)
    return sum(prefill_tokens[:idx + 1])


def _total_prefill(prefill_tokens: list[int]) -> int:
    """Total prefill tokens across all steps."""
    return sum(prefill_tokens) if prefill_tokens else 0


def _run_config(prm, canon_to_batches, emb_dir, data_dir, thresholds,
                support_mode, n_qwen_support=16, rolling_window=3,
                min_completed=0):
    """Run rolling early stopping simulation for one config."""
    rng = np.random.RandomState(42)
    results = {t: [] for t in thresholds}

    eligible = 0
    skipped = 0

    for canon, batch_entries in sorted(canon_to_batches.items()):
        if len(batch_entries) < 2:
            skipped += 1
            continue

        first_label, first_stem = batch_entries[0]

        # Load initial support from first batch
        first_emb = _load_task_embeddings(first_stem, emb_dir)
        first_rewards, _ = _load_task_meta(first_stem, data_dir)
        if first_emb is None or first_rewards is None:
            skipped += 1
            continue

        first_rids = sorted(set(first_emb.keys()) & set(first_rewards.keys()))
        first_labels = [1 if first_rewards[r] >= 0.5 else 0 for r in first_rids]

        if support_mode == "grpo_support_rolling":
            if len(set(first_labels)) < 2:
                skipped += 1
                continue
            sup_act = [first_emb[r][0] for r in first_rids]
            sup_res = [first_emb[r][1] for r in first_rids]
            sup_labels = first_labels
            qwen_act, qwen_res, qwen_labels = [], [], []

        elif support_mode in ("combined_rolling", "combined_rolling_stratified",
                               "best_available_rolling", "best_available_rolling_stratified"):
            use_stratified = support_mode.endswith("_stratified")
            con_result = _load_consolidated_support(canon, n_qwen_support, rng, stratified=use_stratified)
            if support_mode.startswith("combined_rolling") and con_result is None:
                skipped += 1
                continue
            if con_result is not None:
                qwen_act, qwen_res, qwen_labels = con_result
            else:
                qwen_act, qwen_res, qwen_labels = [], [], []

            sup_act = qwen_act + [first_emb[r][0] for r in first_rids]
            sup_res = qwen_res + [first_emb[r][1] for r in first_rids]
            sup_labels = qwen_labels + first_labels

            if len(set(sup_labels)) < 2:
                skipped += 1
                continue
        else:
            raise ValueError(f"Unknown support mode: {support_mode}")

        eligible += 1
        query_entries = batch_entries[1:]

        # Set up rolling queues
        n_qwen = len(qwen_labels)
        grpo_b0_act = sup_act[n_qwen:]
        grpo_b0_res = sup_res[n_qwen:]
        grpo_b0_labels = sup_labels[n_qwen:]

        b0_pass = [(a, r) for a, r, l in zip(grpo_b0_act, grpo_b0_res, grpo_b0_labels) if l == 1]
        b0_fail = [(a, r) for a, r, l in zip(grpo_b0_act, grpo_b0_res, grpo_b0_labels) if l == 0]

        thresh_pass_q = {t: [list(b0_pass)] for t in thresholds}
        thresh_fail_q = {t: [list(b0_fail)] for t in thresholds}

        print(f"  [{eligible}] {canon}: {len(query_entries)} query batches", flush=True)

        for ql, q_stem in query_entries:
            query_emb = _load_task_embeddings(q_stem, emb_dir)
            query_rewards, query_prefill = _load_task_meta(q_stem, data_dir)
            if query_emb is None or query_rewards is None:
                continue

            batch_name = ql.split("/")[0]

            for thresh in thresholds:
                pq = thresh_pass_q[thresh]
                fq = thresh_fail_q[thresh]
                grpo_act = [ar[0] for batch in pq for ar in batch] + \
                           [ar[0] for batch in fq for ar in batch]
                grpo_res = [ar[1] for batch in pq for ar in batch] + \
                           [ar[1] for batch in fq for ar in batch]
                grpo_labels = [1] * sum(len(b) for b in pq) + \
                              [0] * sum(len(b) for b in fq)
                combined_act = qwen_act + grpo_act
                combined_res = qwen_res + grpo_res
                combined_labels = qwen_labels + grpo_labels
                if len(set(combined_labels)) < 2:
                    continue
                prm.set_support(combined_act, combined_res, combined_labels)

                batch_passes = []
                batch_fails = []

                all_rids = sorted(set(query_emb.keys()) & set(query_rewards.keys()))
                for rid_idx, rid in enumerate(all_rids):
                    act_seq, res_seq = query_emb[rid]
                    true_reward = query_rewards[rid]
                    true_label = 1 if true_reward >= 0.5 else 0

                    # min_completed: first N rollouts always run to completion
                    forced_complete = rid_idx < min_completed

                    if forced_complete:
                        early = False
                        decision = true_label
                        effective_reward = true_reward
                        term_step = act_seq.shape[0]
                        total_steps = term_step
                    else:
                        decision, term_step, total_steps = _early_stop_rollout(
                            prm, act_seq, res_seq, thresh)
                        early = term_step < total_steps
                        effective_reward = float(decision) if early else true_reward

                    # Compute token savings
                    rid_prefill = query_prefill.get(rid, []) if query_prefill else []
                    total_tok = _total_prefill(rid_prefill)
                    if early and rid_prefill:
                        used_tok = _prefill_at_step(rid_prefill, term_step)
                    else:
                        used_tok = total_tok

                    if not early:
                        batch_passes.append((act_seq, res_seq)) if true_label == 1 \
                            else batch_fails.append((act_seq, res_seq))

                    results[thresh].append({
                        "task": canon,
                        "batch": batch_name,
                        "rid": rid,
                        "true_label": true_label,
                        "true_reward": true_reward,
                        "decision": decision,
                        "effective_reward": effective_reward,
                        "early": early,
                        "term_step": term_step,
                        "total_steps": total_steps,
                        "total_prefill_tokens": total_tok,
                        "used_prefill_tokens": used_tok,
                    })

                # Rolling eviction
                if batch_passes or batch_fails:
                    pq.append(batch_passes)
                    fq.append(batch_fails)
                    while len(pq) > rolling_window:
                        total_pass = sum(len(b) for b in pq)
                        oldest = pq[0]
                        remaining = total_pass - len(oldest)
                        if remaining >= MIN_CLASS_SUPPORT:
                            pq.pop(0)
                        else:
                            keep = MIN_CLASS_SUPPORT - remaining
                            pq.pop(0)
                            if 0 < keep <= len(oldest):
                                pq.insert(0, oldest[-keep:])
                            elif keep > len(oldest):
                                pq.insert(0, oldest)
                            break
                    while len(fq) > rolling_window:
                        total_fail = sum(len(b) for b in fq)
                        oldest = fq[0]
                        remaining = total_fail - len(oldest)
                        if remaining >= MIN_CLASS_SUPPORT:
                            fq.pop(0)
                        else:
                            keep = MIN_CLASS_SUPPORT - remaining
                            fq.pop(0)
                            if 0 < keep <= len(oldest):
                                fq.insert(0, oldest[-keep:])
                            elif keep > len(oldest):
                                fq.insert(0, oldest)
                            break

    return results, eligible, skipped


def _compute_baseline_prefill(data_dir: Path) -> int:
    """Sum prefill tokens across ALL rollouts (including non-eligible tasks)."""
    total = 0
    for meta_file in sorted(data_dir.glob("batch_*_retail_*.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        for rid, prefill_list in meta.get("prefill_tokens", {}).items():
            total += sum(prefill_list)
    return total


def _compute_metrics(records, baseline_prefill: int):
    """Compute summary metrics using prefill token savings.

    baseline_prefill is the total prefill across ALL rollouts in the run,
    not just the ones where early stopping was applied.
    """
    n = len(records)
    if n == 0:
        return {}

    early_records = [r for r in records if r["early"]]
    n_early = len(early_records)

    early_correct = sum(1 for r in early_records if r["decision"] == r["true_label"])
    overall_correct = sum(1 for r in records
                         if (r["decision"] if r["early"] else
                             (1 if r["true_reward"] >= 0.5 else 0)) == r["true_label"])

    scored_tokens = sum(r["total_prefill_tokens"] for r in records)
    used_tokens = sum(r["used_prefill_tokens"] for r in records)
    tokens_saved = scored_tokens - used_tokens

    total_steps_all = sum(r["total_steps"] for r in records)
    steps_used = sum(min(r["term_step"] + 1, r["total_steps"]) for r in records)
    steps_saved = total_steps_all - steps_used

    actual_total_reward = sum(r["true_reward"] for r in records)
    effective_total_reward = sum(r["effective_reward"] for r in records)
    mae = np.mean([abs(r["effective_reward"] - r["true_reward"]) for r in records])

    return {
        "n_total": n,
        "n_early": n_early,
        "coverage": n_early / n,
        "early_accuracy": early_correct / max(n_early, 1),
        "overall_accuracy": overall_correct / n,
        "baseline_prefill_tokens": baseline_prefill,
        "scored_prefill_tokens": scored_tokens,
        "tokens_saved": tokens_saved,
        "compute_saved": tokens_saved / max(baseline_prefill, 1),
        "compute_saved_scored_only": tokens_saved / max(scored_tokens, 1),
        "total_steps": total_steps_all,
        "steps_saved": steps_saved,
        "step_fraction_saved": steps_saved / max(total_steps_all, 1),
        "actual_mean_reward": actual_total_reward / n,
        "effective_mean_reward": effective_total_reward / n,
        "reward_mae": float(mae),
        "reward_bias": (effective_total_reward - actual_total_reward) / n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Early stopping simulation on official GRPO baselines")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument("--configs", nargs="+",
                        default=["grpo_support_rolling", "combined_rolling",
                                 "combined_rolling_stratified",
                                 "best_available_rolling",
                                 "best_available_rolling_stratified"])
    parser.add_argument("--n-qwen-support", type=int, default=16)
    parser.add_argument("--rolling-window", type=int, default=3)
    parser.add_argument("--min-completed", type=int, default=0,
                        help="Number of rollouts per batch forced to complete (0=all eligible for ES)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prm = ProtoNetPRM.load(args.checkpoint)
    print(f"Loaded PRM from {args.checkpoint}")

    for seed in args.seed:
        seed_dir = GRPO_REPLAY / f"official_retail_s{seed}"
        emb_dir = seed_dir / "embeddings"
        data_dir = seed_dir / "data"

        if not emb_dir.exists():
            print(f"Seed {seed}: embeddings not found at {emb_dir}, skipping")
            continue

        canon_to_batches = _discover_tasks(data_dir)
        print(f"\nSeed {seed}: {len(canon_to_batches)} canonical tasks")

        baseline_prefill = _compute_baseline_prefill(data_dir)
        print(f"Baseline prefill: {baseline_prefill:,} tokens")

        for config in args.configs:
            print(f"\n{'=' * 60}")
            print(f"Seed {seed} | Config: {config}")
            print(f"  Thresholds: {args.thresholds}")
            print(f"{'=' * 60}")

            results, eligible, skipped = _run_config(
                prm, canon_to_batches, emb_dir, data_dir,
                args.thresholds, config, args.n_qwen_support,
                rolling_window=args.rolling_window,
                min_completed=args.min_completed)

            print(f"\n  Eligible tasks: {eligible}, Skipped: {skipped}")

            config_results = {}
            for thresh in args.thresholds:
                records = results[thresh]
                metrics = _compute_metrics(records, baseline_prefill)

                print(f"\n  Threshold={thresh}:")
                print(f"    Overall accuracy: {metrics['overall_accuracy']:.1%}")
                print(f"    Early accuracy:   {metrics['early_accuracy']:.1%}")
                print(f"    Coverage:         {metrics['coverage']:.1%}")
                print(f"    Compute saved:    {metrics['compute_saved']:.1%} "
                      f"(tokens saved: {metrics['tokens_saved']:,} / "
                      f"baseline: {metrics['baseline_prefill_tokens']:,})")
                print(f"    Step fraction:    {metrics['step_fraction_saved']:.1%}")
                print(f"    Reward MAE:       {metrics['reward_mae']:.3f}")
                print(f"    Reward bias:      {metrics['reward_bias']:+.3f}")

                config_results[str(thresh)] = {
                    "summary": metrics,
                }

            mc_suffix = f"_mc{args.min_completed}" if args.min_completed > 0 else ""
            out_path = OUT_DIR / f"results_{config}{mc_suffix}_s{seed}.json"
            with open(out_path, "w") as f:
                json.dump(config_results, f, indent=2, default=str)
            print(f"\n  Saved to {out_path}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Config':<25} {'Seed':>4} {'Thresh':>6} {'Acc':>6} "
          f"{'Cov':>6} {'Saved':>7} {'MAE':>6}")
    print("-" * 70)
    mc_suffix = f"_mc{args.min_completed}" if args.min_completed > 0 else ""
    for seed in args.seed:
        for config in args.configs:
            out_path = OUT_DIR / f"results_{config}{mc_suffix}_s{seed}.json"
            if not out_path.exists():
                continue
            with open(out_path) as f:
                data = json.load(f)
            for thresh in sorted(data.keys(), key=float):
                m = data[thresh]["summary"]
                print(f"{config:<25} s{seed:>2} {float(thresh):>6.2f} "
                      f"{m['overall_accuracy']:>5.1%} {m['coverage']:>5.1%} "
                      f"{m['compute_saved']:>6.1%} {m['reward_mae']:>5.3f}")


if __name__ == "__main__":
    main()
