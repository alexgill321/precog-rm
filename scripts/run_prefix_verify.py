#!/usr/bin/env python3
"""Prefix prediction verification: qwen_t2 only.

Trains ProtoNetPRM in two modes (full-rollout and prefix-augmented),
then evaluates AUROC at rollout fractions 0.05–1.00 on held-out tasks.

Usage:
  cd projects/flow_eval
  python run_prefix_verify.py
"""
from __future__ import annotations

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from _paths import TABLE1, ABL  # noqa: E402  (also puts the repo root on sys.path)
from src.prm.cluster_sequence import ProtoNetPRM
from src.prm.paired_action_response import load_paired_embeddings
from src.prm.embeddings import task_id_to_filename
from collections import defaultdict

CONSOLIDATED = TABLE1 / "tau2_qwen"   # the released tau2/qwen cell (rollouts + manifest + embeddings)


def _build_rollout_emb_sequences(act, res, indices):
    """Group flat (N, D) step arrays into per-rollout (T, D) sequences."""
    ra, rr = defaultdict(list), defaultdict(list)
    for i, (rid, step) in enumerate(indices):
        ra[rid].append((step, act[i])); rr[rid].append((step, res[i]))
    act_seqs = {rid: np.stack([v for _, v in sorted(ra[rid], key=lambda x: x[0])]) for rid in ra}
    res_seqs = {rid: np.stack([v for _, v in sorted(rr[rid], key=lambda x: x[0])]) for rid in rr}
    return act_seqs, res_seqs


def load_all_tasks(domains):
    """Tasks of the given tau2 domains from the released consolidated qwen cell:
    rewards from manifest.json, step embeddings from embeddings/{action,response}/<task>.npz."""
    manifest = json.load((CONSOLIDATED / "manifest.json").open())
    all_tasks = []
    for domain in domains:
        tasks_for_domain = {label: e for label, e in manifest.items() if e["domain"] == domain}
        print(f"Domain '{domain}': {len(tasks_for_domain)} tasks in manifest")
        for label, entry in sorted(tasks_for_domain.items()):
            stem = task_id_to_filename(label)
            act_npz = CONSOLIDATED / "embeddings" / "action" / f"{stem}.npz"
            res_npz = CONSOLIDATED / "embeddings" / "response" / f"{stem}.npz"
            if not (act_npz.exists() and res_npz.exists()):
                print(f"  SKIP {label}: no embedding NPZ found"); continue
            act, res, indices = load_paired_embeddings(str(act_npz), str(res_npz))
            rewards = {r["rid"]: (r["reward"] if r["reward"] is not None else 0.0) for r in entry["rollouts"]}
            rids = sorted(set(r for r, _ in indices) & set(rewards))
            outcome = {r: int(rewards.get(r, 0) >= 0.5) for r in rids}
            act_seqs, res_seqs = _build_rollout_emb_sequences(act, res, indices)
            rids = [r for r in rids if r in act_seqs and r in res_seqs]
            act_seqs = {r: act_seqs[r] for r in rids}; res_seqs = {r: res_seqs[r] for r in rids}
            sr = np.mean([outcome[r] for r in rids]); step_lengths = [act_seqs[r].shape[0] for r in rids]
            all_tasks.append(dict(task_label=label, domain=domain, rids=rids, outcome=outcome, rewards=rewards,
                                  act_seqs=act_seqs, res_seqs=res_seqs, success_rate=float(sr), n_rollouts=len(rids),
                                  n_prefix_examples=sum(2 * act_seqs[r].shape[0] for r in rids),
                                  d_emb=act_seqs[rids[0]].shape[1], steps_min=int(np.min(step_lengths)),
                                  steps_mean=float(np.mean(step_lengths)), steps_max=int(np.max(step_lengths))))
    return all_tasks

FRACTIONS = [round(0.05 * i, 2) for i in range(1, 21)]
SEED = 42
TEST_FRAC = 0.2
K_SUPPORT_EVAL = 8


def is_degenerate(task):
    sr = task["success_rate"]
    return sr <= 0.0 or sr >= 1.0


def is_near_degenerate(task, threshold=0.95):
    sr = task["success_rate"]
    return sr > threshold or sr < (1 - threshold)


def snap_to_fractions(step_scores, fractions=FRACTIONS):
    """Map step-level scores to fraction-indexed scores."""
    T = len(step_scores)
    result = {}
    for f in fractions:
        idx = max(0, int(f * T) - 1)
        idx = min(idx, T - 1)
        result[f] = step_scores[idx]
    return result


def _collect_calibration_data(prm, val_tasks, n_repeats=3,
                              cal_fractions=(0.25, 0.50, 0.75, 1.00)):
    """Score val rollouts at multiple prefix fractions, return logits and labels."""
    all_logits = []
    all_labels = []
    all_fracs = []

    for task in val_tasks:
        rids = task["rids"]
        act_seqs = [task["act_seqs"][r] for r in rids]
        res_seqs = [task["res_seqs"][r] for r in rids]
        labels = [task["outcome"][r] for r in rids]

        if is_degenerate(task):
            continue

        pass_idx = [i for i, l in enumerate(labels) if l == 1]
        fail_idx = [i for i, l in enumerate(labels) if l == 0]
        k = min(K_SUPPORT_EVAL, len(pass_idx) - 1, len(fail_idx) - 1)
        if k < 1:
            continue

        for rep in range(n_repeats):
            rng = np.random.RandomState(SEED + 200 + rep)
            pi, fi = list(pass_idx), list(fail_idx)
            rng.shuffle(pi)
            rng.shuffle(fi)
            support_idx = pi[:k] + fi[:k]
            query_idx = [i for i in range(len(labels)) if i not in set(support_idx)]
            if not query_idx:
                continue

            prm.set_support(
                [act_seqs[i] for i in support_idx],
                [res_seqs[i] for i in support_idx],
                [labels[i] for i in support_idx],
            )

            for qi in query_idx:
                T_steps = act_seqs[qi].shape[0]
                for frac in cal_fractions:
                    l = max(1, int(T_steps * frac))
                    embed = prm._encode_rollout(act_seqs[qi][:l], res_seqs[qi][:l])
                    logits = prm._logits_from_embed(embed)
                    all_logits.append(logits)
                    all_labels.append(labels[qi])
                    all_fracs.append(frac)

    return np.array(all_logits), np.array(all_labels), np.array(all_fracs)


def calibrate_platt(prm, val_tasks, n_repeats=3,
                    cal_fractions=(0.25, 0.50, 0.75, 1.00)):
    """Fit Platt scaling (a, b) on val set: P = sigmoid(a * logit_diff + b).

    logit_diff = -dist_pass - (-dist_fail) = dist_fail - dist_pass.
    Fits across multiple prefix fractions.
    """
    from scipy.optimize import minimize_scalar, minimize

    all_logits, all_labels, all_fracs = _collect_calibration_data(
        prm, val_tasks, n_repeats, cal_fractions)

    logit_diff = all_logits[:, 1] - all_logits[:, 0]

    print(f"  Calibration: {len(all_logits)} samples from {len(val_tasks)} val tasks "
          f"({len(cal_fractions)} fractions × {n_repeats} repeats)")

    def neg_log_likelihood(params):
        a, b = params
        z = a * logit_diff + b
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(all_labels * np.log(p) + (1 - all_labels) * np.log(1 - p))

    result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method="Nelder-Mead")
    a_opt, b_opt = result.x
    loss = result.fun

    # Compare with uncalibrated
    loss_uncal = neg_log_likelihood([1.0, 0.0])

    print(f"  Platt scaling: a={a_opt:.3f}, b={b_opt:.3f}")
    print(f"  Log-loss: {loss_uncal:.4f} (uncalibrated) -> {loss:.4f} (calibrated)")

    return a_opt, b_opt


def _truncate_to_fraction(seq, frac):
    """Truncate a (T, D) array to the first frac*T steps."""
    T = seq.shape[0]
    l = max(1, int(T * frac))
    return seq[:l]


def evaluate_prefix_selective(prm, test_tasks, threshold: float,
                              fractions=FRACTIONS, n_repeats=5,
                              support_fraction=None):
    """At each prefix fraction f, restrict to queries whose score lies
    outside the abstention band (s >= threshold OR s <= 1 - threshold)
    and report selective accuracy + coverage. Same support/query
    protocol as evaluate_prefix_auroc.

    Returns (mean_acc, mean_cov) each dict {f: value or None}.
    """
    per_frac_correct = {f: 0 for f in fractions}
    per_frac_committed = {f: 0 for f in fractions}
    per_frac_total = {f: 0 for f in fractions}

    for task in test_tasks:
        rids = task["rids"]
        act_seqs = [task["act_seqs"][r] for r in rids]
        res_seqs = [task["res_seqs"][r] for r in rids]
        labels = [task["outcome"][r] for r in rids]
        if is_degenerate(task):
            continue
        pass_idx = [i for i, l in enumerate(labels) if l == 1]
        fail_idx = [i for i, l in enumerate(labels) if l == 0]
        k_per_class = min(K_SUPPORT_EVAL, len(pass_idx) - 1, len(fail_idx) - 1)
        if k_per_class < 1:
            continue

        for rep in range(n_repeats):
            rng = np.random.RandomState(SEED + 1 + rep)
            pi, fi = list(pass_idx), list(fail_idx)
            rng.shuffle(pi); rng.shuffle(fi)
            support_idx = pi[:k_per_class] + fi[:k_per_class]
            query_idx = [i for i in range(len(labels)) if i not in set(support_idx)]
            if not query_idx:
                continue

            if support_fraction is not None:
                sup_act = [_truncate_to_fraction(act_seqs[i], support_fraction) for i in support_idx]
                sup_res = [_truncate_to_fraction(res_seqs[i], support_fraction) for i in support_idx]
            else:
                sup_act = [act_seqs[i] for i in support_idx]
                sup_res = [res_seqs[i] for i in support_idx]
            prm.set_support(sup_act, sup_res, [labels[i] for i in support_idx])

            for qi in query_idx:
                scores = prm.score_path(
                    path=list(range(act_seqs[qi].shape[0])),
                    act_emb=act_seqs[qi],
                    res_emb=res_seqs[qi],
                )
                snapped = snap_to_fractions(scores, fractions)
                y = labels[qi]
                for f in fractions:
                    per_frac_total[f] += 1
                    s = snapped[f]
                    pred = None
                    if s >= threshold:
                        pred = 1
                    elif s <= 1 - threshold:
                        pred = 0
                    if pred is not None:
                        per_frac_committed[f] += 1
                        if pred == y:
                            per_frac_correct[f] += 1

    mean_acc = {f: (per_frac_correct[f] / per_frac_committed[f]
                    if per_frac_committed[f] > 0 else None)
                for f in fractions}
    mean_cov = {f: (per_frac_committed[f] / per_frac_total[f]
                    if per_frac_total[f] > 0 else 0.0)
                for f in fractions}
    return mean_acc, mean_cov


def plot_selective_prediction(selective_results: dict, out_path: Path,
                              thresholds_to_plot=None):
    """2-panel: coverage vs fraction, selective accuracy vs fraction.
    One line per (mode, threshold).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    MODE_COLORS = {"full": "#2563eb", "prefix": "#dc2626"}
    MODE_LABELS = {"full": "Full-rollout", "prefix": "Prefix-trained"}
    THR_LS = {0.7: ":", 0.8: "-", 0.85: "--", 0.9: "-."}

    for mode, by_thr in selective_results.items():
        c = MODE_COLORS.get(mode, "#666")
        mlbl = MODE_LABELS.get(mode, mode)
        for thr, d in sorted(by_thr.items()):
            if thresholds_to_plot is not None and thr not in thresholds_to_plot:
                continue
            ls = THR_LS.get(thr, "-")
            acc = d["acc"]; cov = d["coverage"]
            fracs = sorted(f for f in cov if cov[f] is not None)
            cov_y = [cov[f] for f in fracs]
            acc_y = [acc[f] if acc[f] is not None else float("nan") for f in fracs]
            label = f"{mlbl} (θ={thr})"
            axes[0].plot(fracs, cov_y, ls, color=c, linewidth=2, label=label)
            axes[1].plot(fracs, acc_y, ls, color=c, linewidth=2, label=label)

    for ax, ylabel, title in zip(
        axes,
        ["Coverage", "Selective accuracy"],
        ["Fraction of queries committed", "Accuracy among committed"],
    ):
        ax.set_xlabel("Rollout fraction", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0.4, 1.02)

    fig.suptitle("Confidence-Gated Selective Prediction (Qwen τ²)", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


def evaluate_prefix_auroc(prm, test_tasks, fractions=FRACTIONS, n_repeats=5,
                          support_fraction=None):
    """Compute AUROC at each rollout fraction across test tasks.

    For each test task, uses K_SUPPORT_EVAL rollouts per class as support,
    scores the remaining rollouts at each prefix fraction.
    Repeats n_repeats times with different support/query splits and averages.

    If support_fraction is set, truncates support rollouts to that fraction
    before computing prototypes.

    Returns {fraction: mean_auroc} and {fraction: per_task_aurocs}.
    """
    per_frac_aurocs = {f: [] for f in fractions}
    per_frac_accs = {f: [] for f in fractions}

    for task in test_tasks:
        rids = task["rids"]
        act_seqs = [task["act_seqs"][r] for r in rids]
        res_seqs = [task["res_seqs"][r] for r in rids]
        labels = [task["outcome"][r] for r in rids]

        if is_degenerate(task):
            continue

        pass_idx = [i for i, l in enumerate(labels) if l == 1]
        fail_idx = [i for i, l in enumerate(labels) if l == 0]

        k_per_class = min(K_SUPPORT_EVAL, len(pass_idx) - 1, len(fail_idx) - 1)
        if k_per_class < 1:
            continue

        rep_aurocs = {f: [] for f in fractions}
        rep_accs = {f: [] for f in fractions}

        for rep in range(n_repeats):
            rng = np.random.RandomState(SEED + 1 + rep)

            pi = list(pass_idx)
            fi = list(fail_idx)
            rng.shuffle(pi)
            rng.shuffle(fi)
            support_idx = pi[:k_per_class] + fi[:k_per_class]
            query_idx = [i for i in range(len(labels)) if i not in set(support_idx)]

            if not query_idx:
                continue

            query_labels = [labels[i] for i in query_idx]
            if len(set(query_labels)) < 2:
                continue

            if support_fraction is not None:
                sup_act = [_truncate_to_fraction(act_seqs[i], support_fraction) for i in support_idx]
                sup_res = [_truncate_to_fraction(res_seqs[i], support_fraction) for i in support_idx]
            else:
                sup_act = [act_seqs[i] for i in support_idx]
                sup_res = [res_seqs[i] for i in support_idx]

            prm.set_support(sup_act, sup_res, [labels[i] for i in support_idx])

            frac_scores = {f: [] for f in fractions}
            y_true = []

            for qi in query_idx:
                scores = prm.score_path(
                    path=list(range(act_seqs[qi].shape[0])),
                    act_emb=act_seqs[qi],
                    res_emb=res_seqs[qi],
                )
                snapped = snap_to_fractions(scores, fractions)
                for f in fractions:
                    frac_scores[f].append(snapped[f])
                y_true.append(labels[qi])

            y_arr = np.array(y_true)
            for f in fractions:
                s_arr = np.array(frac_scores[f])
                if len(set(y_arr)) >= 2:
                    auroc = float(roc_auc_score(y_arr, s_arr))
                    acc = float(np.mean((s_arr >= 0.5) == y_arr))
                    rep_aurocs[f].append(auroc)
                    rep_accs[f].append(acc)

        for f in fractions:
            if rep_aurocs[f]:
                per_frac_aurocs[f].append(float(np.mean(rep_aurocs[f])))
                per_frac_accs[f].append(float(np.mean(rep_accs[f])))

    mean_aurocs = {}
    mean_accs = {}
    for f in fractions:
        vals = per_frac_aurocs[f]
        mean_aurocs[f] = float(np.mean(vals)) if vals else None
        acc_vals = per_frac_accs[f]
        mean_accs[f] = float(np.mean(acc_vals)) if acc_vals else None

    return mean_aurocs, per_frac_aurocs, mean_accs, per_frac_accs


def simulate_early_stopping(prm, test_tasks, thresholds=None, n_repeats=5):
    """Simulate sequential early stopping across threshold values.

    Scores each rollout once via score_path(), then sweeps thresholds in numpy.
    """
    if thresholds is None:
        thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95

    # Phase 1: score all rollouts once
    scored_rollouts = []

    for task in test_tasks:
        rids = task["rids"]
        act_seqs = [task["act_seqs"][r] for r in rids]
        res_seqs = [task["res_seqs"][r] for r in rids]
        labels = [task["outcome"][r] for r in rids]

        if is_degenerate(task):
            continue

        pass_idx = [i for i, l in enumerate(labels) if l == 1]
        fail_idx = [i for i, l in enumerate(labels) if l == 0]
        k = min(K_SUPPORT_EVAL, len(pass_idx) - 1, len(fail_idx) - 1)
        if k < 1:
            continue

        for rep in range(n_repeats):
            rng = np.random.RandomState(SEED + 1 + rep)
            pi, fi = list(pass_idx), list(fail_idx)
            rng.shuffle(pi)
            rng.shuffle(fi)
            support_idx = pi[:k] + fi[:k]
            query_idx = [i for i in range(len(labels)) if i not in set(support_idx)]
            if not query_idx:
                continue

            prm.set_support(
                [act_seqs[i] for i in support_idx],
                [res_seqs[i] for i in support_idx],
                [labels[i] for i in support_idx],
            )

            for qi in query_idx:
                T = act_seqs[qi].shape[0]
                scores = prm.score_path(
                    path=list(range(T)),
                    act_emb=act_seqs[qi],
                    res_emb=res_seqs[qi],
                )
                scored_rollouts.append({
                    "scores": scores,
                    "true": labels[qi],
                    "total_steps": T,
                    "task": task["task_label"],
                })

    print(f"  Scored {len(scored_rollouts)} rollouts, sweeping {len(thresholds)} thresholds...")

    # Phase 2: sweep thresholds on cached scores
    all_results = {t: [] for t in thresholds}
    for rollout in scored_rollouts:
        scores = rollout["scores"]
        T = rollout["total_steps"]
        true_label = rollout["true"]

        for thresh in thresholds:
            decision = None
            term_step = T
            for t, s in enumerate(scores):
                if s > thresh:
                    decision = 1
                    term_step = t
                    break
                elif (1 - s) > thresh:
                    decision = 0
                    term_step = t
                    break
            if decision is None:
                decision = 1 if scores[-1] >= 0.5 else 0

            all_results[thresh].append({
                "true": true_label,
                "pred": decision,
                "step": term_step,
                "total_steps": T,
                "early": term_step < T,
                "task": rollout["task"],
            })

    metrics = {}
    for thresh in thresholds:
        records = all_results[thresh]
        if not records:
            continue
        n = len(records)
        correct = sum(1 for r in records if r["pred"] == r["true"])
        early = sum(1 for r in records if r["early"])
        compute_saved = np.mean([1 - r["step"] / max(r["total_steps"], 1) for r in records])

        early_records = [r for r in records if r["early"]]
        early_fail_preds = [r for r in early_records if r["pred"] == 0]
        early_pass_preds = [r for r in early_records if r["pred"] == 1]
        true_fails = [r for r in records if r["true"] == 0]

        early_correct = sum(1 for r in early_records if r["pred"] == r["true"])
        early_acc = early_correct / max(len(early_records), 1)

        early_fail_correct = sum(1 for r in early_fail_preds if r["true"] == 0)
        early_pass_correct = sum(1 for r in early_pass_preds if r["true"] == 1)
        precision_fail = early_fail_correct / max(len(early_fail_preds), 1)
        precision_pass = early_pass_correct / max(len(early_pass_preds), 1)
        recall_fail = early_fail_correct / max(len(true_fails), 1)

        early_saved = np.mean([1 - r["step"] / max(r["total_steps"], 1)
                               for r in early_records]) if early_records else 0.0

        metrics[thresh] = {
            "accuracy": correct / n,
            "early_accuracy": early_acc,
            "coverage": early / n,
            "compute_saved": float(compute_saved),
            "compute_saved_early": float(early_saved),
            "precision_fail": precision_fail,
            "precision_pass": precision_pass,
            "recall_fail": recall_fail,
            "n_total": n,
            "n_early": early,
            "n_early_fail": len(early_fail_preds),
            "n_early_pass": len(early_pass_preds),
        }

    return metrics, all_results


def plot_early_stopping(metrics, out_path):
    """Plot early stopping metrics vs threshold."""
    thresholds = sorted(metrics.keys())
    accuracy = [metrics[t]["accuracy"] for t in thresholds]
    early_acc = [metrics[t]["early_accuracy"] for t in thresholds]
    coverage = [metrics[t]["coverage"] for t in thresholds]
    compute_saved = [metrics[t]["compute_saved"] for t in thresholds]
    compute_saved_early = [metrics[t]["compute_saved_early"] for t in thresholds]
    precision_fail = [metrics[t]["precision_fail"] for t in thresholds]
    precision_pass = [metrics[t]["precision_pass"] for t in thresholds]
    recall_fail = [metrics[t]["recall_fail"] for t in thresholds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confidence-Gated Early Stopping", fontsize=13, fontweight="bold")

    # Left: accuracy + precision + coverage
    ax1.plot(thresholds, early_acc, "-o", color="#2563eb", linewidth=2.5,
             markersize=5, label="Early termination accuracy")
    ax1.plot(thresholds, precision_fail, "--s", color="#dc2626", linewidth=1.5,
             markersize=4, label="Precision (early fails)")
    ax1.plot(thresholds, precision_pass, "--^", color="#16a34a", linewidth=1.5,
             markersize=4, label="Precision (early passes)")
    ax1b = ax1.twinx()
    ax1b.bar(thresholds, coverage, width=0.035, alpha=0.15, color="#6b7280",
             label="Coverage")
    ax1.set_xlabel("Confidence Threshold", fontsize=11)
    ax1.set_ylabel("Accuracy / Precision", fontsize=11)
    ax1b.set_ylabel("Coverage", fontsize=11, color="#6b7280")
    ax1b.set_ylim(0, 1.05)
    ax1.set_ylim(0.4, 1.02)
    ax1.legend(loc="lower left", fontsize=9)
    ax1b.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right: compute saved + recall
    ax2.plot(thresholds, compute_saved, "-o", color="#16a34a", linewidth=2,
             markersize=5, label="Compute saved (all rollouts)")
    ax2.plot(thresholds, compute_saved_early, "-s", color="#059669", linewidth=1.5,
             markersize=4, label="Compute saved (terminated only)")
    ax2.plot(thresholds, recall_fail, "--^", color="#d97706", linewidth=2,
             markersize=5, label="Recall (fails caught early)")
    ax2.set_xlabel("Confidence Threshold", fontsize=11)
    ax2.set_ylabel("Fraction", fontsize=11)
    ax2.set_ylim(0, 1.02)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Early stopping plot saved to {out_path}")


PLOT_STYLES = {
    "full":           ("-",  "#2563eb", "Full rollout"),
    "prefix":         ("--", "#dc2626", "Prefix (uniform)"),
    "prefix_0.5":     ("-.", "#16a34a", "Prefix (frac=0.5)"),
    "prefix_0.75":    (":",  "#d97706", "Prefix (frac=0.75)"),
    "prefix_1.0":     ("-.", "#7c3aed", "Prefix (frac=1.0)"),
    "prefix_sup0.5":  ("-.", "#059669", "Prefix (sup=0.5)"),
    "prefix_sup0.75": (":",  "#b45309", "Prefix (sup=0.75)"),
}


def plot_prefix_curves(results, out_path):
    """Plot AUROC vs fraction for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, res in results.items():
        style, color, label = PLOT_STYLES.get(
            mode, ("-", "#666", mode))
        data = res["mean_aurocs"]
        fracs = sorted(f for f in data if data[f] is not None)
        aurocs = [data[f] for f in fracs]
        ax.plot(fracs, aurocs, style, color=color, linewidth=2, label=label)

    ax.set_xlabel("Rollout fraction", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Prefix Prediction: Qwen τ²", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prefix prediction verification (qwen_t2)")
    parser.add_argument("--domains", nargs="+",
                        default=["airline", "retail", "telecom"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--episodes-per-epoch", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Run prefix eval every N epochs during training")
    parser.add_argument("--fixed-fractions", type=float, nargs="*", default=[],
                        help="Additional fixed-fraction prefix models to train (e.g. 0.5 0.75)")
    parser.add_argument("--pooling", default="mean", choices=["mean", "attention"],
                        help="Pooling strategy (default: mean)")
    parser.add_argument("--support-fractions", type=float, nargs="*", default=[],
                        help="Train models with truncated support at these fractions (e.g. 0.5 0.75)")
    parser.add_argument("--early-stopping-plot", action="store_true",
                        help="Run early stopping simulation and plot")
    parser.add_argument("--selective-prediction", action="store_true",
                        help="Run confidence-gated selective prediction for full + prefix models")
    parser.add_argument("--selective-thresholds", type=float, nargs="*",
                        default=[0.7, 0.8, 0.9],
                        help="Confidence thresholds for selective prediction (default: 0.7 0.8 0.9)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Fit temperature scaling on val set before eval")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load saved checkpoints and run eval only")
    args = parser.parse_args()

    default_out = str(ABL / "prefix_verify")
    if args.out_dir:
        out = Path(args.out_dir)
    elif args.calibrate:
        out = Path(default_out + "_calibrated")
    else:
        out = Path(default_out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────
    print("Loading tasks...")
    tasks = load_all_tasks(args.domains)

    # Filter degenerate/near-degenerate
    valid_tasks = []
    for t in tasks:
        if is_degenerate(t) or is_near_degenerate(t):
            print(f"  Skipping {t['task_label']} (degenerate/near-degenerate)")
            continue
        valid_tasks.append(t)

    print(f"{len(valid_tasks)} valid tasks (of {len(tasks)} total)")

    # ── Split tasks ────────────────────────────────────────────────
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(len(valid_tasks))
    n_test = max(1, int(len(valid_tasks) * TEST_FRAC))
    test_indices = sorted(indices[:n_test])
    test_idx = set(test_indices)
    train_tasks = [valid_tasks[i] for i in range(len(valid_tasks))
                   if i not in test_idx]
    test_tasks = [valid_tasks[i] for i in test_indices]

    # Val: hold out 20% of rollouts from each train task
    val_tasks = []
    train_tasks_trimmed = []
    val_rng = np.random.RandomState(SEED + 100)
    for t in train_tasks:
        rids = list(t["rids"])
        val_rng.shuffle(rids)
        n_val = max(1, int(len(rids) * 0.20))
        val_rids = rids[:n_val]
        train_rids = rids[n_val:]

        def _subset(task, rid_list):
            return dict(
                task_label=task["task_label"],
                domain=task["domain"],
                rids=rid_list,
                outcome={r: task["outcome"][r] for r in rid_list},
                rewards={r: task["rewards"][r] for r in rid_list if r in task["rewards"]},
                act_seqs={r: task["act_seqs"][r] for r in rid_list},
                res_seqs={r: task["res_seqs"][r] for r in rid_list},
                success_rate=float(np.mean([task["outcome"][r] for r in rid_list])),
                n_rollouts=len(rid_list),
                d_emb=task["d_emb"],
            )

        train_tasks_trimmed.append(_subset(t, train_rids))
        val_t = _subset(t, val_rids)
        if not is_degenerate(val_t):
            val_tasks.append(val_t)

    train_tasks = train_tasks_trimmed

    print(f"Train: {len(train_tasks)} tasks, Val: {len(val_tasks)} tasks (held-out rollouts), Test: {len(test_tasks)} tasks")

    split_info = {
        "seed": SEED,
        "train_tasks": [t["task_label"] for t in train_tasks],
        "val_tasks": [t["task_label"] for t in val_tasks],
        "test_tasks": [t["task_label"] for t in test_tasks],
    }
    with open(out / "splits.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # ── Prepare training data ──────────────────────────────────────
    all_act, all_res, all_labels, all_task_ids = [], [], [], []
    for t in train_tasks:
        for rid in t["rids"]:
            all_act.append(t["act_seqs"][rid])
            all_res.append(t["res_seqs"][rid])
            all_labels.append(t["outcome"][rid])
            all_task_ids.append(t["task_label"])

    print(f"\nTraining data: {len(all_act)} rollouts across "
          f"{len(train_tasks)} tasks")
    print(f"  Pass rate: {sum(all_labels)/len(all_labels):.1%}")

    # ── Train and evaluate all modes ─────────────────────────────
    skip_main_eval = (args.early_stopping_plot or args.selective_prediction) and args.eval_only

    modes = ["full", "prefix"]
    for frac in args.fixed_fractions:
        modes.append(f"prefix_{frac}")
    for sf in args.support_fractions:
        modes.append(f"prefix_sup{sf}")

    results = {}

    for mode in ([] if skip_main_eval else modes):
        print(f"\n{'=' * 60}")
        print(f"MODE: {mode}")
        print(f"{'=' * 60}")

        ckpt_path = out / f"model_{mode}.pt"

        if args.eval_only and ckpt_path.exists():
            print(f"  Loading checkpoint from {ckpt_path}")
            prm = ProtoNetPRM.load(ckpt_path)
        else:
            prm = ProtoNetPRM(
                epochs=args.epochs,
                episodes_per_epoch=args.episodes_per_epoch,
            )
            prm.pooling = args.pooling
            if mode == "full":
                prm.prefix_training = False
            elif mode == "prefix":
                prm.prefix_training = True
                prm.prefix_fraction = None
            elif mode.startswith("prefix_sup"):
                sf_val = float(mode.split("sup")[1])
                prm.prefix_training = True
                prm.prefix_fraction = sf_val
                prm.support_fraction = sf_val
            else:
                frac_val = float(mode.split("_")[1])
                prm.prefix_training = True
                prm.prefix_fraction = frac_val

            epoch_curves = {}

            def epoch_cb(prm_ref, epoch, loss):
                if (epoch + 1) % args.eval_every != 0:
                    return
                aurocs, _, _, _ = evaluate_prefix_auroc(prm_ref, val_tasks, n_repeats=1)
                epoch_curves[epoch + 1] = aurocs
                auc_25 = aurocs.get(0.25)
                auc_50 = aurocs.get(0.50)
                auc_100 = aurocs.get(1.00)
                print(f"    [eval@{epoch+1}] frac=0.25:{auc_25:.3f}  "
                      f"0.50:{auc_50:.3f}  1.00:{auc_100:.3f}")

            t0 = time.time()
            prm.fit_from_embeddings(
                all_act, all_res, all_labels, all_task_ids,
                epoch_callback=epoch_cb,
            )
            elapsed = time.time() - t0
            print(f"  Training took {elapsed:.0f}s")

            prm.save(ckpt_path)

            if epoch_curves:
                with open(out / f"epoch_curves_{mode}.json", "w") as fh:
                    json.dump({str(k): v for k, v in epoch_curves.items()}, fh, indent=2)

        # Final evaluation
        eval_sf = None
        if mode.startswith("prefix_sup"):
            eval_sf = float(mode.split("sup")[1])
        print(f"\n  Evaluating prefix prediction on {len(test_tasks)} tasks..."
              + (f" (support_fraction={eval_sf})" if eval_sf else ""))
        t0 = time.time()
        mean_aurocs, per_task, mean_accs, per_task_accs = evaluate_prefix_auroc(
            prm, test_tasks, support_fraction=eval_sf)
        elapsed = time.time() - t0
        print(f"  Eval took {elapsed:.0f}s")

        # Print results
        print(f"\n  {'Frac':>6s}  {'AUROC':>7s}  {'Acc':>7s}  {'N tasks':>7s}")
        print(f"  {'-'*32}")
        for f in FRACTIONS:
            v = mean_aurocs[f]
            a = mean_accs[f]
            n = len(per_task[f])
            if v is not None:
                print(f"  {f:>6.2f}  {v:>7.3f}  {a:>6.1%}  {n:>7d}")
            else:
                print(f"  {f:>6.2f}  {'N/A':>7s}  {'N/A':>7s}  {n:>7d}")

        results[mode] = {
            "mean_aurocs": mean_aurocs,
            "mean_accs": mean_accs,
            "per_task_aurocs": {str(f): v for f, v in per_task.items()},
            "per_task_accs": {str(f): v for f, v in per_task_accs.items()},
            "n_test_tasks": len(test_tasks),
        }

        with open(out / f"prefix_eval_{mode}.json", "w") as f:
            json.dump(results[mode], f, indent=2)

    # ── Compare ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("COMPARISON (vs full baseline)")
    print(f"{'=' * 60}")
    header = f"  {'Frac':>6s}  {'Full':>7s}"
    other_modes = [m for m in modes if m != "full"]
    for m in other_modes:
        header += f"  {m:>10s}"
    print(header)
    print(f"  {'-'*(10 + 12 * len(other_modes))}")
    for f in FRACTIONS:
        vf = results.get("full", {}).get("mean_aurocs", {}).get(f)
        row = f"  {f:>6.2f}  {vf:>7.3f}" if vf is not None else f"  {f:>6.2f}  {'N/A':>7s}"
        for m in other_modes:
            vm = results.get(m, {}).get("mean_aurocs", {}).get(f)
            if vm is not None and vf is not None:
                delta = vm - vf
                row += f"  {vm:.3f}({delta:+.3f})"
            elif vm is not None:
                row += f"  {vm:>10.3f}"
            else:
                row += f"  {'N/A':>10s}"
        print(row)

    # ── Confidence-gated selective prediction (full + prefix) ─────
    if args.selective_prediction:
        print(f"\n{'=' * 60}")
        print("SELECTIVE PREDICTION (full + prefix, threshold sweep)")
        print(f"{'=' * 60}")

        selective_results = {}
        for mode in ("full", "prefix"):
            ckpt = out / f"model_{mode}.pt"
            if not ckpt.exists():
                ckpt = Path(default_out) / f"model_{mode}.pt"
            if not ckpt.exists():
                print(f"  [skip] no checkpoint for {mode}")
                continue
            print(f"\n  --- Mode: {mode} (ckpt: {ckpt}) ---")
            prm_sel = ProtoNetPRM.load(ckpt)

            selective_results[mode] = {}
            for thr in args.selective_thresholds:
                print(f"  threshold θ={thr:.2f} ...")
                acc, cov = evaluate_prefix_selective(
                    prm_sel, test_tasks, threshold=thr)
                selective_results[mode][thr] = {"acc": acc, "coverage": cov}

                # Pretty print sample of fractions
                print(f"    {'f':>5s} {'cov':>6s} {'sel_acc':>8s}")
                for f in (0.10, 0.25, 0.50, 0.75, 1.00):
                    c = cov.get(f, 0.0)
                    a = acc.get(f)
                    a_str = f"{a:>7.3f}" if a is not None else "  N/A"
                    print(f"    {f:>5.2f} {c:>5.1%} {a_str}")

        if selective_results:
            with open(out / "selective_prediction.json", "w") as fh:
                json.dump(
                    {m: {str(t): v for t, v in d.items()}
                     for m, d in selective_results.items()},
                    fh, indent=2,
                )
            plot_selective_prediction(
                selective_results, out / "selective_prediction.png")

    # ── Early stopping simulation ─────────────────────────────────
    if args.early_stopping_plot:
        print(f"\n{'=' * 60}")
        print("EARLY STOPPING SIMULATION")
        print(f"{'=' * 60}")

        prefix_ckpt = out / "model_prefix.pt"
        if not prefix_ckpt.exists():
            prefix_ckpt = Path(default_out) / "model_prefix.pt"
        if prefix_ckpt.exists():
            prm_es = ProtoNetPRM.load(prefix_ckpt)
        elif "prefix" in results:
            prm_es = prm  # use last trained model
        else:
            print("  No prefix model available, skipping")
            prm_es = None

        if prm_es is not None:
            if args.calibrate:
                print("\n  Calibrating (Platt scaling) on val set...")
                a_opt, b_opt = calibrate_platt(prm_es, val_tasks)
                prm_es.platt_a = a_opt
                prm_es.platt_b = b_opt
                prm_es.temperature = 1.0
                prm_es.save(out / "model_prefix_calibrated.pt")

            es_metrics, es_raw = simulate_early_stopping(prm_es, test_tasks)

            print(f"\n  {'Thresh':>6} {'Acc':>6} {'EarlyAcc':>8} {'Cov':>6} {'Saved':>6} {'SavedE':>6} {'PrecF':>6} {'PrecP':>6} {'RecF':>6}")
            print(f"  {'-'*62}")
            for t in sorted(es_metrics.keys()):
                m = es_metrics[t]
                print(f"  {t:>6.2f} {m['accuracy']:>5.1%} {m['early_accuracy']:>7.1%} "
                      f"{m['coverage']:>5.1%} {m['compute_saved']:>5.1%} "
                      f"{m['compute_saved_early']:>5.1%} {m['precision_fail']:>5.1%} "
                      f"{m['precision_pass']:>5.1%} {m['recall_fail']:>5.1%}")

            plot_early_stopping(es_metrics, out / "early_stopping_curves.png")

            with open(out / "early_stopping_data.json", "w") as fh:
                json.dump({str(k): v for k, v in es_metrics.items()}, fh, indent=2)

    # ── Plot ───────────────────────────────────────────────────────
    if results:
        plot_prefix_curves(results, out / "prefix_curves.pdf")
        with open(out / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nAll results saved to {out}")


if __name__ == "__main__":
    main()
