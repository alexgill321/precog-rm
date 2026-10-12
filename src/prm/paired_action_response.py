#!/usr/bin/env python3
"""Paired action-response clustering for TDP trajectories.

Embeds each step as an (action, response) pair and clusters using a joint
similarity matrix that requires both action AND response to be similar.
This captures outcome-relevant details (e.g. specific tool arguments) that
thinking-only embeddings miss.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.prm.thought_dag import (
    _build_state_features,
    _cosine_similarity_matrix,
    _discover_tasks,
    _load_rewards_from_files,
    _sr_color,
    build_global_thought_graph,
    render_global_graph,
    render_trajectory_flow,
)


# ── Loading ──────────────────────────────────────────────────────────


def _load_all_embeddings(npz_path: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Load all non-zero embeddings from an NPZ file."""
    data = np.load(npz_path)
    vecs, indices = [], []
    for key in sorted(data.files):
        m = re.match(r"r(\d+)_s(\d+)", key)
        if not m:
            continue
        arr = data[key]
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if np.allclose(arr, 0):
            continue
        vecs.append(arr)
        indices.append((int(m.group(1)), int(m.group(2))))
    return np.stack(vecs).astype(np.float32), indices


def load_paired_embeddings(
    action_npz: str, response_npz: str,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Load action and response embeddings, aligned by (rollout, step) key.

    Only keeps steps present in both NPZs with non-zero vectors.
    Returns (action_matrix, response_matrix, indices).
    """
    act_mat, act_idx = _load_all_embeddings(action_npz)
    res_mat, res_idx = _load_all_embeddings(response_npz)

    act_map = {idx: i for i, idx in enumerate(act_idx)}
    res_map = {idx: i for i, idx in enumerate(res_idx)}

    shared = sorted(set(act_idx) & set(res_idx))

    act_rows = np.stack([act_mat[act_map[k]] for k in shared])
    res_rows = np.stack([res_mat[res_map[k]] for k in shared])

    return act_rows.astype(np.float32), res_rows.astype(np.float32), shared


# ── Paired similarity ───────────────────────────────────────────────


def paired_similarity(
    action_mat: np.ndarray,
    response_mat: np.ndarray,
    method: str = "product",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute joint similarity from action and response embeddings.

    Returns (S_paired, S_action, S_response).
    """
    S_action = _cosine_similarity_matrix(action_mat)
    S_response = _cosine_similarity_matrix(response_mat)

    if method == "product":
        S_paired = S_action * S_response
    elif method == "min":
        S_paired = np.minimum(S_action, S_response)
    elif method == "mean":
        S_paired = 0.5 * S_action + 0.5 * S_response
    else:
        raise ValueError(f"Unknown method: {method}")

    return S_paired, S_action, S_response


# ── Spectral clustering on a precomputed affinity ────────────────────


def spectral_cluster_affinity(
    affinity: np.ndarray, k: int,
) -> np.ndarray:
    """Run spectral clustering given a precomputed affinity matrix."""
    from sklearn.cluster import SpectralClustering

    aff = np.maximum(affinity, 0.0)
    sc = SpectralClustering(
        n_clusters=k, affinity="precomputed",
        random_state=42, n_init=10, assign_labels="kmeans",
    )
    return sc.fit_predict(aff)


def hdbscan_cluster_affinity(
    affinity: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    noise_strategy: str = "nearest",
) -> np.ndarray:
    """Run HDBSCAN on a precomputed affinity (similarity) matrix.

    Converts similarity -> distance, then clusters.

    noise_strategy controls how HDBSCAN outlier points (label -1) are handled:
      - "nearest" / "nearest_neighbor": reassign each outlier to the cluster id
        of its single nearest non-outlier neighbor *point* (not the nearest
        cluster centroid).
      - "outlier_bin": collect all outliers into a single extra cluster K.

    Returns integer labels with no -1, contiguous 0..K-1 (or 0..K when
    outlier_bin adds one extra cluster).
    """
    from sklearn.cluster import HDBSCAN

    dist = 1.0 - np.clip(affinity, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        store_centers=None,
    )
    labels = clusterer.fit_predict(dist)

    noise_mask = labels == -1
    if noise_mask.any() and not noise_mask.all():
        if noise_strategy in ("nearest", "nearest_neighbor"):
            non_noise_idx = np.where(~noise_mask)[0]
            for i in np.where(noise_mask)[0]:
                dists_to_non_noise = dist[i, non_noise_idx]
                nearest = non_noise_idx[np.argmin(dists_to_non_noise)]
                labels[i] = labels[nearest]
        elif noise_strategy in ("noise_bin", "outlier_bin"):
            noise_label = labels.max() + 1
            labels[noise_mask] = noise_label
        else:
            raise ValueError(f"Unknown noise_strategy: {noise_strategy!r}")

    # Relabel to contiguous 0..K-1
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[l] for l in labels])

    return labels


def apply_tool_arg_penalty(
    S: np.ndarray,
    step_tool_info: list[tuple[str, dict] | None],
    beta: float = 2.0,
) -> np.ndarray:
    """Penalize similarity between steps that call the same tool with different args.

    For each pair (i, j) where both call the same tool, computes
    ``arg_sim = n_matching_kv / n_total_keys`` and scales
    ``S[i,j] *= arg_sim ** beta``.  Steps with no tool call or different
    tool names are left unchanged.

    Returns a new matrix; *S* is not modified in place.
    """
    N = S.shape[0]
    assert len(step_tool_info) == N, (
        f"step_tool_info length {len(step_tool_info)} != S dim {N}"
    )

    S_out = S.copy()

    # Group embedding row indices by tool name
    tool_groups: dict[str, list[int]] = defaultdict(list)
    for idx, info in enumerate(step_tool_info):
        if info is not None:
            tool_groups[info[0]].append(idx)

    for _tool_name, row_ids in tool_groups.items():
        if len(row_ids) < 2:
            continue
        args_list = [step_tool_info[i][1] for i in row_ids]  # type: ignore[index]
        m = len(row_ids)

        # Build pairwise arg_sim sub-matrix
        for a in range(m):
            args_a = args_list[a]
            for b in range(a + 1, m):
                args_b = args_list[b]
                all_keys = set(args_a.keys()) | set(args_b.keys())
                if not all_keys:
                    continue
                matching = sum(
                    1
                    for k in all_keys
                    if k in args_a and k in args_b and args_a[k] == args_b[k]
                )
                arg_sim = matching / len(all_keys)
                penalty = arg_sim ** beta
                i, j = row_ids[a], row_ids[b]
                S_out[i, j] *= penalty
                S_out[j, i] *= penalty

    return S_out


def eigengap_k(affinity: np.ndarray, max_k: int = 30) -> int:
    """Estimate K via eigengap heuristic on a precomputed affinity."""
    n = affinity.shape[0]
    max_k = min(max_k, n - 1)
    W = np.maximum(affinity, 0.0)
    np.fill_diagonal(W, 0.0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals = np.sort(np.real(np.linalg.eigvalsh(L)))
    nonzero = eigvals[1 : max_k + 2]
    if len(nonzero) < 2:
        return 2
    gaps = np.diff(nonzero)
    if len(gaps) == 0:
        return 2
    best = int(np.argmax(gaps))
    k = best + 2
    if gaps[best] < 0.05 * nonzero[0]:
        k = 2
    return max(k, 2)


# ── CV-R² prediction ────────────────────────────────────────────────


def _cv_r2(X: np.ndarray, y: np.ndarray, n_folds: int = 5, n_repeats: int = 20) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    if X.shape[0] < n_folds or np.std(y) == 0:
        return 0.0
    scores = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep * 1000 + 42)
        for tr, te in kf.split(X):
            m = Ridge(alpha=1.0).fit(X[tr], y[tr])
            yp = m.predict(X[te])
            ss_res = np.sum((y[te] - yp) ** 2)
            ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
            scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(scores))


def paired_cv_prediction(
    rollout_paths: dict[int, list[int]],
    outcome_map: dict[int, float],
    rewards: dict[int, float],
    k: int,
    label: str = "",
) -> dict:
    """Cross-validated R² prediction using state path features from paired clustering."""
    all_rids = sorted(set(rollout_paths.keys()) & set(outcome_map.keys()))
    y = np.array([outcome_map[r] for r in all_rids])
    if np.std(y) == 0:
        return {"error": "no variance in outcome"}

    X_orig = np.array([[1 if rewards.get(r, 0) >= 0.5 else 0] for r in all_rids]).reshape(-1, 1)
    orig_cv = _cv_r2(X_orig, y)

    feature_sets = _build_state_features(rollout_paths, all_rids, k)
    feature_cv = {name: _cv_r2(X, y) for name, X in feature_sets.items()}

    n_unique = len(set(tuple(rollout_paths[r]) for r in all_rids))
    n_singletons = sum(
        1 for _, cnt in Counter(tuple(rollout_paths[r]) for r in all_rids).items()
        if cnt == 1
    )

    print(f"\n  PAIRED CV R² — {label}")
    print(f"  Rollouts: {len(all_rids)}, Unique paths: {n_unique}, Singletons: {n_singletons}")
    print(f"  Original outcome CV-R²:  {orig_cv:.3f}")
    for name in sorted(feature_cv):
        print(f"  {name:<22s} CV-R²:  {feature_cv[name]:.3f}")

    return {
        "orig_cv": orig_cv,
        "feature_cv": feature_cv,
        "n_rollouts": len(all_rids),
        "n_unique_paths": n_unique,
        "n_singletons": n_singletons,
    }


# ── Plotting ─────────────────────────────────────────────────────────


def plot_cv_comparison(
    cv_results: dict[int, dict],
    output_path: Path,
    title_suffix: str = "",
):
    """Bar chart of CV-R² across K values and feature types."""
    fig, ax = plt.subplots(figsize=(14, 7))

    k_values = sorted(cv_results.keys())
    feature_names = sorted(next(iter(cv_results.values()))["feature_cv"].keys())

    colors_map = {
        "visit_fractions": "#3498db",
        "states_visited": "#e67e22",
        "bigrams": "#2ecc71",
        "positional": "#9b59b6",
        "combined": "#e74c3c",
    }

    n_features = len(feature_names)
    n_k = len(k_values)
    width = 0.8 / (n_features + 1)
    x = np.arange(n_k)

    for j, fname in enumerate(feature_names):
        vals = [cv_results[k]["feature_cv"].get(fname, 0) for k in k_values]
        color = colors_map.get(fname, f"C{j}")
        ax.bar(x + j * width, vals, width, label=fname, color=color, edgecolor="white")

    orig_vals = [cv_results[k]["orig_cv"] for k in k_values]
    ax.bar(x + n_features * width, orig_vals, width, label="original outcome",
           color="#555555", edgecolor="white")

    ax.set_xticks(x + (n_features / 2) * width)
    k_labels = [f"K={k}\n({cv_results[k]['n_unique_paths']} paths)" for k in k_values]
    ax.set_xticklabels(k_labels, fontsize=9)
    ax.set_ylabel("Cross-validated R²", fontsize=12)
    ax.set_title(f"Paired Action-Response Prediction{title_suffix}", fontsize=13)
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_ylim(-0.3, 1.05)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(
    results_by_method: dict[str, dict[int, dict]],
    output_path: Path,
    title_suffix: str = "",
):
    """Compare product/min/mean combination methods across K."""
    fig, ax = plt.subplots(figsize=(14, 7))

    method_colors = {"product": "#e74c3c", "min": "#3498db", "mean": "#2ecc71"}

    for method, cv_results in results_by_method.items():
        k_values = sorted(cv_results.keys())
        combined_vals = [cv_results[k]["feature_cv"].get("combined", 0) for k in k_values]
        color = method_colors.get(method, "gray")
        ax.plot(k_values, combined_vals, "o-", color=color, lw=2, ms=8, label=f"{method}")

    if cv_results:
        first = next(iter(results_by_method.values()))
        k_values = sorted(first.keys())
        orig = [first[k]["orig_cv"] for k in k_values]
        ax.plot(k_values, orig, "k--", lw=2, label="original outcome")

    ax.set_xlabel("K (number of clusters)", fontsize=12)
    ax.set_ylabel("Combined CV-R²", fontsize=12)
    ax.set_title(f"Pairing Method Comparison{title_suffix}", fontsize=13)
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_task_comparison(
    all_results: dict[str, dict],
    thinking_results_path: str | None,
    out: Path,
):
    """Side-by-side paired vs thinking CV-R² across tasks."""
    tasks = sorted(all_results.keys())
    n = len(tasks)

    short_labels = []
    for t in tasks:
        parts = t.split("/")
        domain = parts[0][:3].upper()
        tid = parts[1] if len(parts) > 1 else t
        if len(tid) > 15:
            tid = tid[:12] + "..."
        short_labels.append(f"{domain}/{tid}")

    paired_cvs = [all_results[t]["best_combined_cv_r2"] for t in tasks]

    thinking_cvs = None
    if thinking_results_path and Path(thinking_results_path).exists():
        with open(thinking_results_path) as f:
            thinking_data = json.load(f)
        thinking_cvs = []
        for t in tasks:
            tdata = thinking_data.get(t, {})
            thinking_cvs.append(tdata.get("best_combined_cv_r2", 0))

    domain_colors = {"airline": "#3498db", "retail": "#e67e22", "telecom": "#2ecc71"}
    colors = [domain_colors.get(all_results[t]["domain"], "gray") for t in tasks]

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    # Panel 1: Paired CV-R² per task
    ax1 = axes[0]
    x = np.arange(n)
    width = 0.35

    bars1 = ax1.bar(x - width / 2, paired_cvs, width, color=colors, edgecolor="white",
                     label="Paired (action+response)")
    if thinking_cvs is not None:
        faded = [(0.7, 0.7, 0.7)] * n
        ax1.bar(x + width / 2, thinking_cvs, width, color=faded, edgecolor="white",
                label="Thinking only", alpha=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Best Combined CV-R²", fontsize=11)
    ax1.set_title("Paired vs Thinking: Predictive Power per Task", fontsize=13)
    ax1.axhline(0, color="gray", ls="-", lw=0.5)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="y")

    for i, (pcv, t) in enumerate(zip(paired_cvs, tasks)):
        bk = all_results[t].get('best_k') or '?'
        ax1.text(i - width / 2, pcv + 0.02,
                 f"K={bk}", ha="center", va="bottom", fontsize=7)

    # Panel 2: Improvement scatter
    ax2 = axes[1]
    if thinking_cvs is not None:
        for i, t in enumerate(tasks):
            ax2.scatter(thinking_cvs[i], paired_cvs[i], s=120, c=colors[i],
                        edgecolors="black", linewidths=0.5, zorder=5)
            ax2.annotate(short_labels[i], (thinking_cvs[i], paired_cvs[i]),
                         fontsize=7, textcoords="offset points", xytext=(5, 5))
        lim_lo = min(min(thinking_cvs), min(paired_cvs)) - 0.1
        lim_hi = max(max(thinking_cvs), max(paired_cvs)) + 0.1
        ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", alpha=0.3, label="y=x")
        ax2.set_xlabel("Thinking-only CV-R²", fontsize=11)
        ax2.set_ylabel("Paired CV-R²", fontsize=11)
        ax2.set_title("Paired vs Thinking per Task", fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
    else:
        srs = [all_results[t]["success_rate"] for t in tasks]
        for i, t in enumerate(tasks):
            ax2.scatter(srs[i], paired_cvs[i], s=120, c=colors[i],
                        edgecolors="black", linewidths=0.5, zorder=5)
            ax2.annotate(short_labels[i], (srs[i], paired_cvs[i]),
                         fontsize=7, textcoords="offset points", xytext=(5, 5))
        ax2.set_xlabel("Task Success Rate", fontsize=11)
        ax2.set_ylabel("Best Combined CV-R²", fontsize=11)
        ax2.set_title("Success Rate vs Predictive Power", fontsize=13)
        ax2.grid(alpha=0.3)
        ax2.axhline(0, color="gray", ls="-", lw=0.5)

    plt.suptitle("Paired Action-Response Clustering — Cross-Task", fontsize=15,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out / "cross_task_paired_vs_thinking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / 'cross_task_paired_vs_thinking.png'}")


# ── Single-task analysis ─────────────────────────────────────────────


def run_single_task(
    action_npz: str,
    response_npz: str,
    rewards: dict[int, float],
    out_dir: Path,
    task_label: str = "",
    k_values: list[int] | None = None,
    methods: list[str] | None = None,
) -> dict:
    """Run paired clustering analysis on one task."""
    out_dir.mkdir(parents=True, exist_ok=True)

    act_mat, res_mat, indices = load_paired_embeddings(action_npz, response_npz)
    print(f"  {act_mat.shape[0]} paired steps (action dim={act_mat.shape[1]}, response dim={res_mat.shape[1]})")

    outcome_map = {r: float(rewards.get(r, 0)) for r in rewards}
    n_pass = sum(1 for r in rewards.values() if r >= 0.5)
    sr = n_pass / len(rewards) if rewards else 0

    if methods is None:
        methods = ["product", "min", "mean"]

    if k_values is None:
        k_values = [3, 5, 8, 10, 15, 20]

    best_overall_k = None
    best_overall_cv = -999.0
    best_method = None
    results_by_method: dict[str, dict[int, dict]] = {}

    for method in methods:
        S_paired, S_action, S_response = paired_similarity(act_mat, res_mat, method)

        eg_k = eigengap_k(S_paired)
        sweep_ks = sorted(set([eg_k] + k_values))

        print(f"\n  Method: {method} (eigengap K={eg_k})")

        method_results: dict[int, dict] = {}
        for k in sweep_ks:
            if k < 2 or k >= act_mat.shape[0]:
                continue
            labels = spectral_cluster_affinity(S_paired, k)
            G, rollout_paths = build_global_thought_graph(labels, indices, rewards)
            cv = paired_cv_prediction(rollout_paths, outcome_map, rewards, k,
                                      f"{task_label} {method} K={k}")
            method_results[k] = cv

            if "error" in cv:
                continue
            combined = cv["feature_cv"].get("combined", -999)
            if combined > best_overall_cv:
                best_overall_cv = combined
                best_overall_k = k
                best_method = method

        results_by_method[method] = method_results

    if best_method is None:
        best_method = methods[0]
        best_overall_k = k_values[0] if k_values else 5
        print(f"\n  No valid CV (constant outcome?) — using default method={best_method}, K={best_overall_k}")
    else:
        print(f"\n  BEST: method={best_method}, K={best_overall_k}, combined CV-R²={best_overall_cv:.3f}")

    # Final outputs for best config
    S_paired, _, _ = paired_similarity(act_mat, res_mat, best_method)
    labels = spectral_cluster_affinity(S_paired, best_overall_k)
    G, rollout_paths = build_global_thought_graph(labels, indices, rewards)

    render_global_graph(G, out_dir / "paired_graph.png",
                        f"Paired Action-Response States — {task_label} (K={best_overall_k}, {best_method})")

    render_trajectory_flow(
        labels, indices, rewards,
        out_dir / "paired_trajectory_flow.png",
        f"Paired Trajectory Flow — {task_label} (K={best_overall_k}, {best_method})",
    )

    # Filter out error entries before plotting
    valid_results = {k: v for k, v in results_by_method.get(best_method, {}).items()
                     if "error" not in v}
    if valid_results:
        plot_cv_comparison(
            valid_results, out_dir / "paired_cv_prediction.png",
            f"\n{task_label} ({best_method})",
        )

    valid_by_method = {m: {k: v for k, v in kr.items() if "error" not in v}
                       for m, kr in results_by_method.items()}
    valid_by_method = {m: kr for m, kr in valid_by_method.items() if kr}
    if valid_by_method:
        plot_method_comparison(
            valid_by_method, out_dir / "paired_method_comparison.png",
            f"\n{task_label}",
        )

    state_counts = Counter(int(l) for l in labels)
    metrics = {
        "task_label": task_label,
        "success_rate": sr,
        "n_paired_steps": act_mat.shape[0],
        "best_method": best_method,
        "best_k": best_overall_k,
        "best_combined_cv_r2": best_overall_cv,
        "state_counts": {str(s): c for s, c in state_counts.items()},
        "all_method_results": {
            method: {
                str(kv): (
                    {"error": v["error"]} if "error" in v else {
                        "orig_cv": v["orig_cv"],
                        "feature_cv": v["feature_cv"],
                        "n_unique_paths": v["n_unique_paths"],
                        "n_singletons": v["n_singletons"],
                    }
                )
                for kv, v in kresults.items()
            }
            for method, kresults in results_by_method.items()
        },
    }
    with open(out_dir / "paired_clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved to {out_dir}")
    return metrics


# ── Multi-task runner ────────────────────────────────────────────────


def run_multi_task(
    data_dir: str,
    action_embed_dir: str,
    response_embed_dir: str,
    out_dir: str,
    thinking_summary: str | None = None,
    k_values: list[int] | None = None,
):
    """Run paired analysis on all tasks and produce cross-task comparison."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Discovering tasks...")
    tasks = _discover_tasks(data_dir, action_embed_dir)
    print(f"  Found {len(tasks)} tasks")

    all_results: dict[str, dict] = {}

    for ti, task in enumerate(tasks):
        label = task["short_id"]
        task_out = out / task["task_label"].replace("/", "_")

        print(f"\n{'#'*70}")
        print(f"# Task {ti+1}/{len(tasks)}: {label}")
        print(f"{'#'*70}")

        rewards = _load_rewards_from_files(task["tdp_files"])
        n_pass = sum(1 for r in rewards.values() if r >= 0.5)
        sr = n_pass / len(rewards) if rewards else 0
        print(f"  {len(rewards)} rollouts, SR={sr:.0%}")

        # Find matching NPZ in response dir
        action_npz = task["npz_path"]
        npz_name = Path(action_npz).name
        response_npz = str(Path(response_embed_dir) / npz_name)

        if not Path(response_npz).exists():
            print(f"  WARNING: No response NPZ at {response_npz}, skipping")
            continue

        metrics = run_single_task(
            action_npz=action_npz,
            response_npz=response_npz,
            rewards=rewards,
            out_dir=task_out,
            task_label=label,
            k_values=k_values,
        )
        metrics["domain"] = task["domain"]
        metrics["task_id"] = task["task_id"]
        all_results[label] = metrics

    # Cross-task summary
    print(f"\n{'='*70}")
    print("CROSS-TASK SUMMARY (Paired Action-Response)")
    print(f"{'='*70}")

    with open(out / "all_tasks_paired_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    plot_cross_task_comparison(all_results, thinking_summary, out)

    print(f"\n{'Task':<45s} {'Domain':<10s} {'SR':>5s} {'Method':<8s} {'K':>3s} {'CV-R²':>7s}")
    print("-" * 85)
    for t in sorted(all_results.keys()):
        r = all_results[t]
        bk = r.get('best_k') or 0
        bcv = r.get('best_combined_cv_r2', -999)
        print(f"{t:<45s} {r['domain']:<10s} {r['success_rate']:>5.0%} "
              f"{r['best_method']:<8s} {bk:>3d} {bcv:>7.3f}")

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Paired action-response clustering")
    parser.add_argument("--data-dir", required=True, help="Experiment data directory")
    parser.add_argument("--action-embed-dir", default=None,
                        help="Action embeddings dir (default: {data-dir}/embeddings/action)")
    parser.add_argument("--response-embed-dir", default=None,
                        help="Response embeddings dir (default: {data-dir}/embeddings/response)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--thinking-summary", default=None,
                        help="Path to thinking-only all_tasks_summary.json for comparison")
    args = parser.parse_args()

    action_dir = args.action_embed_dir or str(Path(args.data_dir) / "embeddings" / "action")
    response_dir = args.response_embed_dir or str(Path(args.data_dir) / "embeddings" / "response")

    run_multi_task(
        data_dir=args.data_dir,
        action_embed_dir=action_dir,
        response_embed_dir=response_dir,
        out_dir=args.out_dir,
        thinking_summary=args.thinking_summary,
    )


if __name__ == "__main__":
    main()
