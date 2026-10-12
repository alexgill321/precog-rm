#!/usr/bin/env python3
"""Build a DAG of thought-cluster states from spectral clustering results.

Nodes are (step, cluster_id) pairs. Edges are rollout transitions between
consecutive steps, weighted by count. Supports divergence analysis and
path-based outcome prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ── DAG construction ──


def load_rewards(data_dir: str, task_id: str) -> dict[int, float]:
    """Load rollout rewards from TDP JSON files."""
    import glob

    rewards: dict[int, float] = {}
    parts = task_id.split("/")
    pattern = f"{data_dir}/{parts[0]}/{parts[1]}_tdp_*.json"
    for fp in glob.glob(pattern):
        idx = int(fp.split("_")[-1].replace(".json", ""))
        with open(fp) as fh:
            rewards[idx] = json.load(fh).get("reward", 0.0)
    return rewards


def build_dag(
    metrics: list[dict],
    rewards: dict[int, float],
    collapse_k1: bool = False,
) -> tuple[nx.DiGraph, dict[int, dict[int, int]]]:
    """Build a networkx DiGraph from cluster_metrics.json entries.

    Returns (graph, step_assignments) where step_assignments maps
    step -> {rollout_id -> cluster_label}.
    """
    G = nx.DiGraph()

    step_assignments: dict[int, dict[int, int]] = {}
    steps_data: list[dict] = []

    for entry in sorted(metrics, key=lambda e: e["step"]):
        step = entry["step"]
        k = entry["k"]
        rids = entry["rollout_ids"]
        labels = entry["labels"]

        if collapse_k1 and k == 1:
            continue

        assignment = dict(zip(rids, labels))
        step_assignments[step] = assignment
        steps_data.append(entry)

        clusters: dict[int, list[int]] = defaultdict(list)
        for rid, lbl in zip(rids, labels):
            clusters[lbl].append(rid)

        for cl, members in clusters.items():
            node_id = (step, cl)
            n = len(members)
            n_pass = sum(1 for r in members if rewards.get(r, 0) >= 0.5)
            sr = n_pass / n if n > 0 else 0.0
            G.add_node(
                node_id,
                step=step,
                cluster=cl,
                size=n,
                success_rate=sr,
                n_pass=n_pass,
                n_fail=n - n_pass,
                members=members,
            )

    sorted_steps = sorted(step_assignments.keys())
    for i in range(len(sorted_steps) - 1):
        s_from = sorted_steps[i]
        s_to = sorted_steps[i + 1]
        a_from = step_assignments[s_from]
        a_to = step_assignments[s_to]

        edge_counts: Counter[tuple[int, int]] = Counter()
        edge_pass: Counter[tuple[int, int]] = Counter()
        for rid in set(a_from) & set(a_to):
            key = (a_from[rid], a_to[rid])
            edge_counts[key] += 1
            if rewards.get(rid, 0) >= 0.5:
                edge_pass[key] += 1

        for (cl_from, cl_to), count in edge_counts.items():
            n_from = (s_from, cl_from)
            n_to = (s_to, cl_to)
            sr = edge_pass[(cl_from, cl_to)] / count if count > 0 else 0.0
            G.add_edge(n_from, n_to, weight=count, success_rate=sr)

    return G, step_assignments


# ── Entropy / information gain ──


def _binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def compute_node_entropy(G: nx.DiGraph) -> dict[tuple, float]:
    ent = {}
    for node, data in G.nodes(data=True):
        sr = data["success_rate"]
        ent[node] = _binary_entropy(sr)
    return ent


def compute_edge_info_gain(G: nx.DiGraph) -> dict[tuple, float]:
    """For each edge, compute information gain: parent entropy - child entropy."""
    node_ent = compute_node_entropy(G)
    gains = {}
    for u, v in G.edges():
        gains[(u, v)] = node_ent[u] - node_ent[v]
    return gains


# ── Path extraction and prediction ──


def extract_rollout_paths(
    step_assignments: dict[int, dict[int, int]],
) -> dict[int, list[tuple[int, int]]]:
    """For each rollout, extract its path: list of (step, cluster_id)."""
    all_rids = set()
    for a in step_assignments.values():
        all_rids.update(a.keys())

    paths: dict[int, list[tuple[int, int]]] = {}
    sorted_steps = sorted(step_assignments.keys())
    for rid in all_rids:
        path = []
        for step in sorted_steps:
            cl = step_assignments[step].get(rid)
            if cl is not None:
                path.append((step, cl))
        paths[rid] = path
    return paths


def path_prediction_analysis(
    paths: dict[int, list[tuple[int, int]]],
    rewards: dict[int, float],
    step_assignments: dict[int, dict[int, int]],
) -> dict:
    """Compare predictive power of path features vs single-step clusters."""
    sorted_steps = sorted(step_assignments.keys())
    all_rids = sorted(set(paths.keys()) & set(rewards.keys()))
    y = np.array([1 if rewards[r] >= 0.5 else 0 for r in all_rids])
    ss_total = np.sum((y - y.mean()) ** 2)
    if ss_total == 0:
        return {"error": "no variance in outcomes"}

    results = {"steps": sorted_steps, "n_rollouts": len(all_rids)}

    # Single-step R²
    step_r2 = {}
    for step in sorted_steps:
        a = step_assignments[step]
        groups: dict[int, list[float]] = defaultdict(list)
        for rid in all_rids:
            cl = a.get(rid)
            if cl is not None:
                groups[cl].append(1.0 if rewards[rid] >= 0.5 else 0.0)
        if len(groups) < 2:
            step_r2[step] = 0.0
            continue
        ss_within = sum(
            np.sum((np.array(v) - np.mean(v)) ** 2) for v in groups.values()
        )
        step_r2[step] = max(0.0, 1 - ss_within / ss_total)
    results["single_step_r2"] = step_r2

    # Path-prefix R² (path up to step S as group key)
    prefix_r2 = {}
    for s_idx, step in enumerate(sorted_steps):
        groups: dict[tuple, list[float]] = defaultdict(list)
        for rid in all_rids:
            path = paths[rid]
            prefix = tuple(
                (st, cl) for st, cl in path if st <= step
            )
            groups[prefix].append(1.0 if rewards[rid] >= 0.5 else 0.0)
        if len(groups) < 2:
            prefix_r2[step] = 0.0
            continue
        ss_within = sum(
            np.sum((np.array(v) - np.mean(v)) ** 2)
            for v in groups.values()
            if len(v) > 0
        )
        prefix_r2[step] = max(0.0, 1 - ss_within / ss_total)
    results["path_prefix_r2"] = prefix_r2

    # Full-path R²
    groups_full: dict[tuple, list[float]] = defaultdict(list)
    for rid in all_rids:
        key = tuple(paths[rid])
        groups_full[key].append(1.0 if rewards[rid] >= 0.5 else 0.0)
    if len(groups_full) >= 2:
        ss_w = sum(
            np.sum((np.array(v) - np.mean(v)) ** 2)
            for v in groups_full.values()
            if len(v) > 0
        )
        results["full_path_r2"] = max(0.0, 1 - ss_w / ss_total)
    else:
        results["full_path_r2"] = 0.0

    results["n_unique_paths"] = len(groups_full)

    return results


# ── Visualization ──


def _sr_color(sr: float) -> str:
    """Map success rate [0,1] to red-yellow-green color."""
    cmap = plt.cm.RdYlGn
    return mcolors.to_hex(cmap(sr))


def render_dag_graphviz(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Thought-State DAG",
):
    """Render the DAG using graphviz (dot)."""
    import graphviz

    dot = graphviz.Digraph(
        comment=title,
        format="png",
        graph_attr={
            "rankdir": "LR",
            "label": title,
            "labelloc": "t",
            "fontsize": "18",
            "dpi": "150",
            "nodesep": "0.4",
            "ranksep": "1.2",
        },
        node_attr={"shape": "ellipse", "style": "filled"},
        edge_attr={"color": "#666666"},
    )

    sorted_steps = sorted(set(data["step"] for _, data in G.nodes(data=True)))

    for step in sorted_steps:
        with dot.subgraph(name=f"cluster_step_{step}") as sub:
            sub.attr(label=f"Step {step}", style="dashed", color="gray")
            for node, data in G.nodes(data=True):
                if data["step"] != step:
                    continue
                sr = data["success_rate"]
                n = data["size"]
                cl = data["cluster"]
                color = _sr_color(sr)
                font_color = "black" if 0.3 < sr < 0.7 else "white"
                label = f"C{cl}\nn={n}\n{sr:.0%}"
                node_name = f"s{step}_c{cl}"
                width = str(max(0.5, min(2.5, 0.3 + n * 0.02)))
                sub.node(
                    node_name,
                    label=label,
                    fillcolor=color,
                    fontcolor=font_color,
                    width=width,
                    height=width,
                    fontsize="10",
                )

    max_weight = max(
        (d["weight"] for _, _, d in G.edges(data=True)), default=1
    )
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        sr = data["success_rate"]
        u_name = f"s{u[0]}_c{u[1]}"
        v_name = f"s{v[0]}_c{v[1]}"
        penwidth = str(max(0.5, min(6.0, 0.5 + 5.0 * w / max_weight)))
        color = _sr_color(sr)
        label = f"{w}" if w >= 3 else ""
        dot.edge(
            u_name, v_name,
            penwidth=penwidth,
            color=color,
            label=label,
            fontsize="8",
        )

    dot.render(str(output_path.with_suffix("")), cleanup=True)


def render_dag_matplotlib(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Thought-State DAG",
):
    """Fallback renderer using matplotlib + networkx layout."""
    sorted_steps = sorted(set(data["step"] for _, data in G.nodes(data=True)))
    step_to_x = {s: i for i, s in enumerate(sorted_steps)}

    pos = {}
    for step in sorted_steps:
        nodes_at_step = [
            n for n, d in G.nodes(data=True) if d["step"] == step
        ]
        nodes_at_step.sort(key=lambda n: n[1])
        n_nodes = len(nodes_at_step)
        for j, node in enumerate(nodes_at_step):
            x = step_to_x[step]
            y = (j - (n_nodes - 1) / 2) * 1.5
            pos[node] = (x, y)

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_steps) * 2.5), 8))

    node_colors = [_sr_color(G.nodes[n]["success_rate"]) for n in G.nodes()]
    node_sizes = [max(200, G.nodes[n]["size"] * 30) for n in G.nodes()]

    max_weight = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    edge_widths = [
        max(0.5, 5.0 * d["weight"] / max_weight)
        for _, _, d in G.edges(data=True)
    ]
    edge_colors = [_sr_color(d["success_rate"]) for _, _, d in G.edges(data=True)]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.5,
    )

    labels = {
        n: f"C{d['cluster']}\nn={d['size']}\n{d['success_rate']:.0%}"
        for n, d in G.nodes(data=True)
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    edge_labels = {
        (u, v): str(d["weight"])
        for u, v, d in G.edges(data=True)
        if d["weight"] >= 3
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, font_size=6, ax=ax,
    )

    for step in sorted_steps:
        x = step_to_x[step]
        ax.text(x, ax.get_ylim()[1] + 0.3, f"Step {step}",
                ha="center", fontsize=10, fontweight="bold")

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.RdYlGn, norm=mcolors.Normalize(0, 1),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Success Rate", fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_dag(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Thought-State DAG",
):
    try:
        import graphviz as _gv  # noqa: F401
        render_dag_graphviz(G, output_path, title)
    except (ImportError, Exception):
        render_dag_matplotlib(G, output_path, title)


# ── Divergence analysis plot ──


def plot_divergence(
    G: nx.DiGraph,
    output_path: Path,
    title_suffix: str = "",
):
    node_ent = compute_node_entropy(G)
    edge_ig = compute_edge_info_gain(G)

    sorted_steps = sorted(set(d["step"] for _, d in G.nodes(data=True)))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: Node entropy by step
    ax1 = axes[0]
    for step in sorted_steps:
        nodes = [(n, d) for n, d in G.nodes(data=True) if d["step"] == step]
        for node, data in nodes:
            color = _sr_color(data["success_rate"])
            ax1.scatter(
                step, node_ent[node], s=data["size"] * 10,
                color=color, edgecolors="black", linewidth=0.5,
                alpha=0.8, zorder=5,
            )
            if node_ent[node] > 0.1 or data["size"] > 15:
                ax1.annotate(
                    f"C{data['cluster']}\n({data['size']})",
                    (step, node_ent[node]),
                    fontsize=6, ha="center",
                    textcoords="offset points", xytext=(0, 10),
                )
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Outcome Entropy (bits)", fontsize=12)
    ax1.set_title(f"Node Entropy{title_suffix}", fontsize=13)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.set_xticks(sorted_steps)

    # Panel 2: Edge information gain
    ax2 = axes[1]
    gains_list = []
    for (u, v), ig in edge_ig.items():
        w = G.edges[u, v]["weight"]
        gains_list.append((u, v, ig, w))
    gains_list.sort(key=lambda x: -abs(x[2]))

    if gains_list:
        labels_ig = [
            f"S{u[0]}C{u[1]}→\nS{v[0]}C{v[1]}"
            for u, v, _, _ in gains_list[:20]
        ]
        values = [ig for _, _, ig, _ in gains_list[:20]]
        weights = [w for _, _, _, w in gains_list[:20]]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]
        bars = ax2.barh(
            range(len(labels_ig)), values,
            color=colors, edgecolor="white", height=0.7,
        )
        ax2.set_yticks(range(len(labels_ig)))
        ax2.set_yticklabels(labels_ig, fontsize=7)
        for i, (bar, w) in enumerate(zip(bars, weights)):
            ax2.text(
                bar.get_width() + 0.01 * (1 if bar.get_width() >= 0 else -1),
                i, f"n={w}", va="center", fontsize=7,
            )
    ax2.set_xlabel("Information Gain (bits)", fontsize=12)
    ax2.set_title(f"Edge Information Gain{title_suffix}", fontsize=13)
    ax2.axvline(0, color="black", lw=0.5)
    ax2.invert_yaxis()

    # Panel 3: Predictive leverage — rank nodes by mutual info with outcome
    ax3 = axes[2]
    leverage = []
    total_n = sum(d["size"] for _, d in G.nodes(data=True))
    base_sr = (
        sum(d["n_pass"] for _, d in G.nodes(data=True))
        / total_n
        if total_n > 0
        else 0.5
    )
    base_sr = sum(
        d["n_pass"]
        for _, d in G.nodes(data=True)
        if d["step"] == sorted_steps[0]
    ) / sum(
        d["size"]
        for _, d in G.nodes(data=True)
        if d["step"] == sorted_steps[0]
    )
    base_ent = _binary_entropy(base_sr)

    for step in sorted_steps:
        nodes_at = [(n, d) for n, d in G.nodes(data=True) if d["step"] == step]
        total_at = sum(d["size"] for _, d in nodes_at)
        if total_at == 0 or len(nodes_at) < 2:
            leverage.append((step, 0.0))
            continue
        weighted_ent = sum(
            (d["size"] / total_at) * _binary_entropy(d["success_rate"])
            for _, d in nodes_at
        )
        mi = base_ent - weighted_ent
        leverage.append((step, mi))

    steps_lev = [s for s, _ in leverage]
    vals_lev = [v for _, v in leverage]
    colors_lev = ["#3498db" if v > 0.05 else "#bdc3c7" for v in vals_lev]
    ax3.bar(steps_lev, vals_lev, color=colors_lev, edgecolor="white")
    ax3.set_xlabel("Step", fontsize=12)
    ax3.set_ylabel("Mutual Information (bits)", fontsize=12)
    ax3.set_title(f"Cluster Predictive Leverage{title_suffix}", fontsize=13)
    ax3.set_xticks(sorted_steps)
    for s, v in zip(steps_lev, vals_lev):
        if v > 0.01:
            ax3.text(s, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_path_prediction(
    pred_results: dict,
    output_path: Path,
    title_suffix: str = "",
):
    steps = pred_results["steps"]
    step_r2 = pred_results["single_step_r2"]
    prefix_r2 = pred_results["path_prefix_r2"]
    full_r2 = pred_results["full_path_r2"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(steps))

    ax.plot(
        x, [step_r2[s] for s in steps],
        "o-", color="#e74c3c", label="Single-step cluster R²", linewidth=2,
    )
    ax.plot(
        x, [prefix_r2[s] for s in steps],
        "s-", color="#2ecc71", label="Path-prefix R² (up to step)", linewidth=2,
    )
    ax.axhline(
        full_r2, color="#3498db", ls="--", lw=2,
        label=f"Full-path R² = {full_r2:.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Step {s}" for s in steps], rotation=45, ha="right")
    ax.set_ylabel("R² (variance explained)", fontsize=12)
    ax.set_title(
        f"Path vs Single-Step Prediction{title_suffix}\n"
        f"({pred_results['n_unique_paths']} unique paths, "
        f"{pred_results['n_rollouts']} rollouts)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──


def run_analysis(
    metrics_path: str,
    data_dir: str,
    task_id: str,
    out_dir: str,
    label: str = "",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(metrics_path) as f:
        metrics = json.load(f)
    rewards = load_rewards(data_dir, task_id)

    suffix = f" ({label})" if label else ""

    # Full DAG
    G_full, assignments_full = build_dag(metrics, rewards, collapse_k1=False)
    print(f"Full DAG{suffix}: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    render_dag(G_full, out / f"dag_full{('_' + label) if label else ''}.png", f"Thought-State DAG{suffix}")

    # Collapsed DAG (skip k=1 steps)
    G_col, assignments_col = build_dag(metrics, rewards, collapse_k1=True)
    print(f"Collapsed DAG{suffix}: {G_col.number_of_nodes()} nodes, {G_col.number_of_edges()} edges")
    render_dag(G_col, out / f"dag_collapsed{('_' + label) if label else ''}.png", f"Thought-State DAG (collapsed){suffix}")

    # Divergence analysis
    plot_divergence(G_col, out / f"divergence{('_' + label) if label else ''}.png", suffix)
    print(f"Divergence analysis{suffix} saved.")

    # Path prediction (use collapsed DAG assignments to skip k=1 noise)
    paths = extract_rollout_paths(assignments_col)
    pred = path_prediction_analysis(paths, rewards, assignments_col)
    plot_path_prediction(pred, out / f"path_prediction{('_' + label) if label else ''}.png", suffix)

    print(f"\nPath prediction results{suffix}:")
    print(f"  Unique paths: {pred['n_unique_paths']}")
    print(f"  Full-path R²: {pred['full_path_r2']:.3f}")
    print(f"  Single-step R² by step:")
    for step in pred["steps"]:
        s_r2 = pred["single_step_r2"][step]
        p_r2 = pred["path_prefix_r2"][step]
        delta = p_r2 - s_r2
        print(f"    Step {step:>2d}: single={s_r2:.3f}  prefix={p_r2:.3f}  delta={delta:+.3f}")

    return G_full, G_col, pred


def main():
    parser = argparse.ArgumentParser(description="Build thought-state DAG from clustering results")
    parser.add_argument("--data-dir", required=True, help="Experiment data directory")
    parser.add_argument("--task-id", default="airline/12", help="Task identifier")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots")
    parser.add_argument("--non-dtw-metrics", default=None, help="Path to non-DTW cluster_metrics.json")
    parser.add_argument("--dtw-metrics", default=None, help="Path to DTW cluster_metrics.json")
    parser.add_argument("--multi-task", action="store_true",
                        help="Run global analysis on all tasks found in embed-dir")
    parser.add_argument("--embed-dir", default=None,
                        help="Embeddings directory (for --multi-task mode)")
    parser.add_argument("--branch-dir", default=None,
                        help="Branch results directory (used for airline/12)")
    args = parser.parse_args()

    if args.multi_task:
        embed_dir = args.embed_dir
        if not embed_dir:
            embed_dir = str(Path(args.data_dir) / "embeddings" / "thinking")
        run_multi_task_analysis(
            data_dir=args.data_dir,
            embed_dir=embed_dir,
            out_dir=args.out_dir,
            branch_dir=args.branch_dir,
        )
        return

    results = {}

    if args.non_dtw_metrics:
        print("=" * 60)
        print("NON-DTW CLUSTERING")
        print("=" * 60)
        results["non_dtw"] = run_analysis(
            args.non_dtw_metrics, args.data_dir, args.task_id, args.out_dir, "non_dtw",
        )

    if args.dtw_metrics:
        print("\n" + "=" * 60)
        print("DTW-ALIGNED CLUSTERING")
        print("=" * 60)
        results["dtw"] = run_analysis(
            args.dtw_metrics, args.data_dir, args.task_id, args.out_dir, "dtw",
        )

    if len(results) == 2:
        print("\n" + "=" * 60)
        print("COMPARISON: NON-DTW vs DTW")
        print("=" * 60)
        _, _, pred_nd = results["non_dtw"]
        _, _, pred_dtw = results["dtw"]
        print(f"  Full-path R²:  non-DTW={pred_nd['full_path_r2']:.3f}  DTW={pred_dtw['full_path_r2']:.3f}")
        print(f"  Unique paths:  non-DTW={pred_nd['n_unique_paths']}  DTW={pred_dtw['n_unique_paths']}")

        fig, ax = plt.subplots(figsize=(10, 6))
        steps_nd = pred_nd["steps"]
        steps_dtw = pred_dtw["steps"]
        x_nd = np.arange(len(steps_nd))
        x_dtw = np.arange(len(steps_dtw))

        ax.plot(x_nd, [pred_nd["path_prefix_r2"][s] for s in steps_nd],
                "s-", color="#e74c3c", label="Non-DTW path-prefix R²", linewidth=2)
        ax.plot(x_dtw, [pred_dtw["path_prefix_r2"][s] for s in steps_dtw],
                "D-", color="#2ecc71", label="DTW path-prefix R²", linewidth=2)
        ax.axhline(pred_nd["full_path_r2"], color="#e74c3c", ls="--", lw=1.5, alpha=0.6,
                   label=f"Non-DTW full R²={pred_nd['full_path_r2']:.3f}")
        ax.axhline(pred_dtw["full_path_r2"], color="#2ecc71", ls="--", lw=1.5, alpha=0.6,
                   label=f"DTW full R²={pred_dtw['full_path_r2']:.3f}")
        ax.set_ylabel("R²", fontsize=12)
        ax.set_title("Path Prediction: Non-DTW vs DTW", fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        max_len = max(len(steps_nd), len(steps_dtw))
        ax.set_xticks(range(max_len))
        tick_labels = []
        for i in range(max_len):
            parts = []
            if i < len(steps_nd):
                parts.append(f"nd:{steps_nd[i]}")
            if i < len(steps_dtw):
                parts.append(f"dtw:{steps_dtw[i]}")
            tick_labels.append("\n".join(parts))
        ax.set_xticklabels(tick_labels, fontsize=8)

        plt.tight_layout()
        fig.savefig(Path(args.out_dir) / "comparison_path_prediction.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Comparison plot saved.")

    print(f"\nAll outputs in: {args.out_dir}")


# ══════════════════════════════════════════════════════════════════════
# Global thought-state graph: step-free, embedding-clustered states
# ══════════════════════════════════════════════════════════════════════


def _load_all_embeddings(
    npz_path: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Load all embeddings from an NPZ file, returning a matrix and index list.

    Returns (matrix [N x dim], index [(rollout_id, step), ...]).
    """
    import re

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


def _cosine_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = vecs / norms
    sim = normed @ normed.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return sim


def _eigengap_k_global(sim: np.ndarray, max_k: int = 30) -> tuple[int, np.ndarray]:
    """Eigengap heuristic for global clustering."""
    n = sim.shape[0]
    max_k = min(max_k, n - 1)
    W = np.maximum(sim, 0.0)
    np.fill_diagonal(W, 0.0)
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals = np.sort(np.real(np.linalg.eigvalsh(L)))

    nonzero_eigs = eigvals[1 : max_k + 2]
    if len(nonzero_eigs) < 2:
        return 1, eigvals

    gaps = np.diff(nonzero_eigs)
    if len(gaps) == 0:
        return 1, eigvals

    best_gap_idx = int(np.argmax(gaps))
    k = best_gap_idx + 2

    if gaps[best_gap_idx] < 0.05 * nonzero_eigs[0]:
        k = 1

    return max(k, 1), eigvals


def global_spectral_cluster(
    matrix: np.ndarray,
    k: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Run spectral clustering on all embeddings.

    Returns (labels, k, eigenvalues).
    """
    from sklearn.cluster import SpectralClustering

    sim = _cosine_similarity_matrix(matrix)

    if k is None:
        k, eigvals = _eigengap_k_global(sim)
    else:
        _, eigvals = _eigengap_k_global(sim)

    if k <= 1:
        return np.zeros(len(matrix), dtype=int), 1, eigvals

    affinity = np.maximum(sim, 0.0)
    np.fill_diagonal(affinity, 0.0)
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        random_state=0,
        assign_labels="kmeans",
    )
    labels = sc.fit_predict(affinity)
    return labels, k, eigvals


def build_global_thought_graph(
    labels: np.ndarray,
    indices: list[tuple[int, int]],
    rewards: dict[int, float],
    branch_rates: dict[tuple[int, int], float] | None = None,
) -> tuple[nx.DiGraph, dict[int, list[int]]]:
    """Build a directed graph where nodes are global thought states.

    Returns (graph, rollout_paths) where rollout_paths maps
    rollout_id -> [state_0, state_1, ...] (sequence of global states).
    """
    G = nx.DiGraph()

    rid_step_to_state = {}
    for (rid, step), state in zip(indices, labels):
        rid_step_to_state[(rid, step)] = int(state)

    all_rids = sorted(set(rid for rid, _ in indices))

    rollout_paths: dict[int, list[int]] = {}
    for rid in all_rids:
        steps_for_rid = sorted(s for r, s in indices if r == rid)
        path = [rid_step_to_state[(rid, s)] for s in steps_for_rid]
        rollout_paths[rid] = path

    unique_states = sorted(set(int(l) for l in labels))
    for state in unique_states:
        visiting_rids = set()
        visit_count = 0
        pass_visits = 0
        branch_sr_values = []
        for (rid, step), s in zip(indices, labels):
            if int(s) == state:
                visiting_rids.add(rid)
                visit_count += 1
                if rewards.get(rid, 0) >= 0.5:
                    pass_visits += 1
                if branch_rates and (step, rid) in branch_rates:
                    branch_sr_values.append(branch_rates[(step, rid)])

        sr = pass_visits / visit_count if visit_count else 0.0
        n_pass = sum(1 for r in visiting_rids if rewards.get(r, 0) >= 0.5)
        mean_branch_sr = float(np.mean(branch_sr_values)) if branch_sr_values else None

        G.add_node(
            state,
            visits=visit_count,
            n_rollouts=len(visiting_rids),
            success_rate=sr,
            n_pass=n_pass,
            n_fail=len(visiting_rids) - n_pass,
            mean_branch_sr=mean_branch_sr,
            n_branch_obs=len(branch_sr_values),
        )

    G.add_node("START", visits=len(all_rids), n_rollouts=len(all_rids),
               success_rate=sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5) / len(all_rids),
               n_pass=sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5),
               n_fail=sum(1 for r in all_rids if rewards.get(r, 0) < 0.5),
               mean_branch_sr=None, n_branch_obs=0)
    G.add_node("END_PASS", visits=0, n_rollouts=0, success_rate=1.0,
               n_pass=0, n_fail=0, mean_branch_sr=None, n_branch_obs=0)
    G.add_node("END_FAIL", visits=0, n_rollouts=0, success_rate=0.0,
               n_pass=0, n_fail=0, mean_branch_sr=None, n_branch_obs=0)

    edge_counts: Counter = Counter()
    edge_pass_counts: Counter = Counter()

    for rid in all_rids:
        path = rollout_paths[rid]
        passed = rewards.get(rid, 0) >= 0.5

        first_state = path[0]
        edge_counts[("START", first_state)] += 1
        if passed:
            edge_pass_counts[("START", first_state)] += 1

        for i in range(len(path) - 1):
            edge_counts[(path[i], path[i + 1])] += 1
            if passed:
                edge_pass_counts[(path[i], path[i + 1])] += 1

        last_state = path[-1]
        end_node = "END_PASS" if passed else "END_FAIL"
        edge_counts[(last_state, end_node)] += 1

    for (src, dst), count in edge_counts.items():
        sr = edge_pass_counts[(src, dst)] / count if count > 0 else 0.0
        G.add_edge(src, dst, weight=count, success_rate=sr)

    n_pass_total = sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5)
    G.nodes["END_PASS"]["visits"] = n_pass_total
    G.nodes["END_PASS"]["n_rollouts"] = n_pass_total
    G.nodes["END_FAIL"]["visits"] = len(all_rids) - n_pass_total
    G.nodes["END_FAIL"]["n_rollouts"] = len(all_rids) - n_pass_total

    return G, rollout_paths


def render_global_graph_graphviz(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Global Thought-State Graph",
):
    """Render the global thought-state graph using Graphviz."""
    import graphviz

    dot = graphviz.Digraph(
        comment=title,
        format="png",
        engine="dot",
        graph_attr={
            "label": title,
            "labelloc": "t",
            "fontsize": "20",
            "dpi": "150",
            "nodesep": "0.6",
            "ranksep": "1.0",
            "overlap": "false",
            "splines": "true",
        },
        node_attr={"style": "filled", "shape": "circle"},
        edge_attr={"color": "#666666"},
    )

    special = {"START", "END_PASS", "END_FAIL"}
    max_visits = max(
        (G.nodes[n]["visits"] for n in G.nodes if n not in special), default=1
    )

    dot.node(
        "START", "START", shape="doublecircle",
        fillcolor="#3498db", fontcolor="white", width="0.8",
    )
    n_pass = G.nodes["END_PASS"]["visits"]
    n_fail = G.nodes["END_FAIL"]["visits"]
    dot.node(
        "END_PASS", f"PASS\n(n={n_pass})", shape="doublecircle",
        fillcolor="#27ae60", fontcolor="white", width="0.8",
    )
    dot.node(
        "END_FAIL", f"FAIL\n(n={n_fail})", shape="doublecircle",
        fillcolor="#e74c3c", fontcolor="white", width="0.8",
    )

    for node in sorted(n for n in G.nodes if n not in special):
        data = G.nodes[node]
        sr = data["success_rate"]
        v = data["visits"]
        nr = data["n_rollouts"]
        color = _sr_color(sr)
        font_color = "black" if 0.3 < sr < 0.7 else "white"
        branch_line = ""
        if data.get("mean_branch_sr") is not None:
            branch_line = f"\nbSR={data['mean_branch_sr']:.0%}"
        label = f"S{node}\n{v} visits, {nr} rollouts\n{sr:.0%}{branch_line}"
        width = str(max(0.6, min(2.5, 0.4 + 2.0 * v / max_visits)))
        dot.node(
            str(node), label=label,
            fillcolor=color, fontcolor=font_color,
            width=width, height=width, fontsize="9",
        )

    max_weight = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    for u, v, data in G.edges(data=True):
        w = data["weight"]
        sr = data["success_rate"]
        penwidth = str(max(0.5, min(8.0, 0.5 + 7.0 * w / max_weight)))
        color = _sr_color(sr)
        u_name = str(u)
        v_name = str(v)
        label = f"{w}" if w >= 2 else ""
        dot.edge(u_name, v_name, penwidth=penwidth, color=color, label=label, fontsize="8")

    dot.render(str(output_path.with_suffix("")), cleanup=True)


def render_global_graph_matplotlib(
    G: nx.DiGraph,
    output_path: Path,
    title: str = "Global Thought-State Graph",
):
    """Fallback renderer for global graph using matplotlib."""
    fig, ax = plt.subplots(figsize=(14, 10))

    special = {"START", "END_PASS", "END_FAIL"}
    regular_nodes = [n for n in G.nodes if n not in special]

    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)

    node_colors = []
    node_sizes = []
    for n in G.nodes:
        data = G.nodes[n]
        if n == "START":
            node_colors.append("#3498db")
            node_sizes.append(800)
        elif n == "END_PASS":
            node_colors.append("#27ae60")
            node_sizes.append(800)
        elif n == "END_FAIL":
            node_colors.append("#e74c3c")
            node_sizes.append(800)
        else:
            node_colors.append(_sr_color(data["success_rate"]))
            node_sizes.append(max(200, data["visits"] * 5))

    max_weight = max((d["weight"] for _, _, d in G.edges(data=True)), default=1)
    edge_widths = [max(0.3, 5.0 * d["weight"] / max_weight) for _, _, d in G.edges(data=True)]
    edge_colors = [_sr_color(d["success_rate"]) for _, _, d in G.edges(data=True)]

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors,
                           alpha=0.6, arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                           edgecolors="black", linewidths=0.5)

    labels = {}
    for n, d in G.nodes(data=True):
        if n in special:
            labels[n] = str(n)
        else:
            labels[n] = f"S{n}\n{d['visits']}v\n{d['success_rate']:.0%}"
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Success Rate", fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_global_graph(G, output_path, title="Global Thought-State Graph"):
    try:
        import graphviz as _gv  # noqa: F401
        render_global_graph_graphviz(G, output_path, title)
    except (ImportError, Exception):
        render_global_graph_matplotlib(G, output_path, title)


# ── Trajectory flow (Sankey / alluvial) visualization ──


def render_trajectory_flow(
    labels: np.ndarray,
    indices: list[tuple[int, int]],
    rewards: dict[int, float],
    output_path: Path | None = None,
    title: str = "Trajectory Flow Through States",
    collapse_uniform: bool = True,
    noise_label: int | None = None,
    highlight_rids: list[int] | None = None,
    return_fig: bool = False,
    min_active_frac: float = 0.0,
):
    """Render a Sankey/alluvial-style flow showing rollouts propagating through
    global cluster states over time.

    X-axis = step (temporal progression), columns = stacked cluster bars,
    ribbons = rollout transitions between consecutive steps.

    If collapse_uniform is True, consecutive steps where all rollouts occupy
    the same single cluster are collapsed into one representative column,
    keeping the diagram compact for long trajectories.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    rid_step_to_state: dict[tuple[int, int], int] = {}
    for (rid, step), state in zip(indices, labels):
        rid_step_to_state[(rid, step)] = int(state)

    all_rids = sorted(set(rid for rid, _ in indices))
    raw_steps = sorted(set(step for _, step in indices))
    unique_states = sorted(set(int(l) for l in labels))

    # Optionally truncate steps where too few rollouts are still active.
    n_rollouts = len(all_rids)
    if min_active_frac > 0:
        min_active = max(1, int(n_rollouts * min_active_frac))
        active_counts = {}
        for step in raw_steps:
            active_counts[step] = len(set(
                rid for rid in all_rids if (rid, step) in rid_step_to_state
            ))
        cutoff_step = None
        for step in raw_steps:
            if active_counts[step] < min_active:
                cutoff_step = step
                break
        if cutoff_step is not None:
            raw_steps = [s for s in raw_steps if s < cutoff_step]
            if not raw_steps:
                raw_steps = sorted(set(step for _, step in indices))[:1]

    # Optionally collapse uniform runs
    if collapse_uniform and len(raw_steps) > 15:
        keep_steps = []
        prev_state_set: set[int] | None = None
        run_start: int | None = None
        for step in raw_steps:
            states_at = set(
                rid_step_to_state[(rid, step)]
                for rid in all_rids
                if (rid, step) in rid_step_to_state
            )
            if len(states_at) == 1 and states_at == prev_state_set:
                continue  # skip, same uniform state
            else:
                if run_start is not None and prev_state_set is not None and len(prev_state_set) == 1:
                    pass  # already added the start of the uniform run
                keep_steps.append(step)
                if len(states_at) == 1:
                    run_start = step
                else:
                    run_start = None
                prev_state_set = states_at
        # Always include the last step
        if raw_steps[-1] not in keep_steps:
            keep_steps.append(raw_steps[-1])
        all_steps = keep_steps
    else:
        all_steps = raw_steps

    # Assign a persistent hue to each state for outline/identity cues
    n_states = len(unique_states)
    state_base_colors = {}
    palette = plt.cm.Set2 if n_states <= 8 else plt.cm.tab20
    for i, s in enumerate(unique_states):
        state_base_colors[s] = mcolors.to_hex(palette(i / max(1, n_states)))

    # Per-step: which rollouts are in which state
    step_state_rids: dict[int, dict[int, list[int]]] = {}
    for step in all_steps:
        groups: dict[int, list[int]] = defaultdict(list)
        for rid in all_rids:
            if (rid, step) in rid_step_to_state:
                groups[rid_step_to_state[(rid, step)]].append(rid)
        step_state_rids[step] = groups

    def _step_state_sr(rids_list):
        if not rids_list:
            return 0.5
        n_pass = sum(1 for r in rids_list if rewards.get(r, 0) >= 0.5)
        return n_pass / len(rids_list)

    # Transition counts between consecutive steps
    transitions: dict[tuple[int, int], dict[tuple[int, int], dict]] = {}
    for si in range(len(all_steps) - 1):
        s_from, s_to = all_steps[si], all_steps[si + 1]
        edge_data: dict[tuple[int, int], dict] = {}
        for rid in all_rids:
            if (rid, s_from) in rid_step_to_state and (rid, s_to) in rid_step_to_state:
                st_from = rid_step_to_state[(rid, s_from)]
                st_to = rid_step_to_state[(rid, s_to)]
                key = (st_from, st_to)
                if key not in edge_data:
                    edge_data[key] = {"total": 0, "pass": 0, "rids": []}
                edge_data[key]["total"] += 1
                edge_data[key]["rids"].append(rid)
                if rewards.get(rid, 0) >= 0.5:
                    edge_data[key]["pass"] += 1
        transitions[(s_from, s_to)] = edge_data

    # ── Layout geometry (adaptive to number of displayed steps) ──
    n_steps = len(all_steps)
    if n_steps <= 15:
        col_width, col_gap = 0.32, 0.88
    elif n_steps <= 25:
        col_width, col_gap = 0.28, 0.65
    else:
        col_width, col_gap = 0.24, 0.50
    bar_pad = 0.06
    total_height = 10.0

    fig_width = max(16, min(50, n_steps * (col_width + col_gap) + 3))
    fig_height = max(7, 9)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_facecolor("#fafafa")

    x_centers = {step: i * (col_width + col_gap) for i, step in enumerate(all_steps)}

    # For each step, compute stacked bar layout (bottom y, height for each state)
    # Use consistent state ordering (by state index) across all steps
    # Scale heights relative to the peak rollout count so columns shrink
    # as rollouts terminate, making attrition visible.
    max_rollouts = max(
        sum(len(rids) for rids in step_state_rids[step].values())
        for step in all_steps
    )
    step_bar_layout: dict[int, dict[int, tuple[float, float]]] = {}
    for step in all_steps:
        groups = step_state_rids[step]
        total_at_step = sum(len(rids) for rids in groups.values())
        if total_at_step == 0:
            continue
        active_states = sorted(groups.keys())
        n_gaps = max(0, len(active_states) - 1)
        scale = total_at_step / max_rollouts
        usable = total_height * scale - n_gaps * bar_pad
        y_cursor = 0.0
        layout: dict[int, tuple[float, float]] = {}
        for state in active_states:
            h = usable * len(groups[state]) / total_at_step
            layout[state] = (y_cursor, h)
            y_cursor += h + bar_pad
        step_bar_layout[step] = layout

    # ── Draw ribbons (flows between steps) ──
    for si in range(len(all_steps) - 1):
        s_from, s_to = all_steps[si], all_steps[si + 1]
        if s_from not in step_bar_layout or s_to not in step_bar_layout:
            continue
        layout_from = step_bar_layout[s_from]
        layout_to = step_bar_layout[s_to]
        edge_data = transitions.get((s_from, s_to), {})

        src_cursors: dict[int, float] = {
            state: bot for state, (bot, _) in layout_from.items()
        }
        dst_cursors: dict[int, float] = {
            state: bot for state, (bot, _) in layout_to.items()
        }

        x_right = x_centers[s_from] + col_width / 2
        x_left = x_centers[s_to] - col_width / 2
        cp_offset = (x_left - x_right) * 0.45

        sorted_edges = sorted(edge_data.items(), key=lambda x: -x[1]["total"])

        for (st_from, st_to), edata in sorted_edges:
            if st_from not in layout_from or st_to not in layout_to:
                continue
            n = edata["total"]
            n_pass = edata["pass"]
            sr = n_pass / n if n > 0 else 0.5

            _, h_from = layout_from[st_from]
            _, h_to = layout_to[st_to]
            total_from = len(step_state_rids[s_from].get(st_from, []))
            total_to = len(step_state_rids[s_to].get(st_to, []))

            ribbon_h_from = h_from * n / max(1, total_from)
            ribbon_h_to = h_to * n / max(1, total_to)

            y_bot_from = src_cursors[st_from]
            y_bot_to = dst_cursors[st_to]

            y_top_from = y_bot_from + ribbon_h_from
            y_top_to = y_bot_to + ribbon_h_to

            # Smooth cubic Bezier ribbon
            verts = [
                (x_right, y_bot_from),
                (x_right + cp_offset, y_bot_from),
                (x_left - cp_offset, y_bot_to),
                (x_left, y_bot_to),
                (x_left, y_top_to),
                (x_left - cp_offset, y_top_to),
                (x_right + cp_offset, y_top_from),
                (x_right, y_top_from),
                (x_right, y_bot_from),
            ]
            codes = [
                MPath.MOVETO,
                MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                MPath.LINETO,
                MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                MPath.CLOSEPOLY,
            ]
            path = MPath(verts, codes)
            is_noise_ribbon = noise_label is not None and (
                st_from == noise_label or st_to == noise_label)
            color = "#8888cc" if is_noise_ribbon else _sr_color(sr)
            alpha = 0.25 if is_noise_ribbon else (
                0.35 + 0.15 * min(1.0, n / max(1, len(all_rids) * 0.3)))
            patch = PathPatch(
                path, facecolor=color, edgecolor=color,
                linewidth=0.3, alpha=alpha, zorder=0 if is_noise_ribbon else 1,
            )
            ax.add_patch(patch)

            src_cursors[st_from] = y_top_from
            dst_cursors[st_to] = y_top_to

    # ── Draw stacked bars (columns) ──
    for step in all_steps:
        if step not in step_bar_layout:
            continue
        layout = step_bar_layout[step]
        groups = step_state_rids[step]
        xc = x_centers[step]

        for state, (y_bot, h) in layout.items():
            rids_in = groups.get(state, [])
            sr = _step_state_sr(rids_in)
            is_noise = noise_label is not None and state == noise_label
            color = _sr_color(sr)
            edge_color = "#2244cc" if is_noise else "black"
            edge_width = 2.2 if is_noise else 0.9
            rect = plt.Rectangle(
                (xc - col_width / 2, y_bot), col_width, h,
                facecolor=color, edgecolor=edge_color,
                linewidth=edge_width, zorder=3,
                hatch="///" if is_noise else None,
            )
            ax.add_patch(rect)

            n = len(rids_in)
            if not is_noise:
                if h > 0.8:
                    font_color = "black" if 0.3 < sr < 0.7 else "white"
                    label = f"S{state}\n{sr:.0%}"
                    ax.text(
                        xc, y_bot + h / 2, label,
                        ha="center", va="center", fontsize=7,
                        fontweight="bold", color=font_color, zorder=4,
                    )
                elif h > 0.4:
                    font_color = "black" if 0.3 < sr < 0.7 else "white"
                    ax.text(
                        xc, y_bot + h / 2, f"S{state}",
                        ha="center", va="center", fontsize=5.5,
                        color=font_color, zorder=4,
                    )

    # ── Step labels and rollout counts ──
    label_fontsize = 9 if n_steps <= 20 else 7
    for step in all_steps:
        xc = x_centers[step]
        total = sum(len(rids) for rids in step_state_rids[step].values())
        ax.text(
            xc, -0.5, f"Step {step}",
            ha="center", va="top", fontsize=label_fontsize, fontweight="bold",
            rotation=45 if n_steps > 20 else 0,
        )
        ax.text(
            xc, -0.9 if n_steps <= 20 else -1.1, f"({total}r)",
            ha="center", va="top", fontsize=6, color="#666666",
        )

    # ── Terminal state indicators ──
    # Show where rollouts terminate (appear in step s but not s+1)
    last_step = all_steps[-1]
    for si in range(len(all_steps) - 1):
        s_from, s_to = all_steps[si], all_steps[si + 1]
        groups_from = step_state_rids[s_from]
        groups_to = step_state_rids[s_to]
        rids_at_from = set()
        for rids in groups_from.values():
            rids_at_from.update(rids)
        rids_at_to = set()
        for rids in groups_to.values():
            rids_at_to.update(rids)
        terminated = rids_at_from - rids_at_to
        if terminated and len(terminated) >= 2:
            xc = x_centers[s_from]
            n_term = len(terminated)
            n_pass = sum(1 for r in terminated if rewards.get(r, 0) >= 0.5)
            ax.text(
                xc, total_height + 0.25,
                f"↓{n_term} exit ({n_pass}✓/{n_term - n_pass}✗)",
                ha="center", va="bottom", fontsize=6, color="#888888",
            )

    # ── Colorbar ──
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.015, location="right")
    cbar.set_label("Success Rate", fontsize=10)

    # ── Outlier legend ──
    if noise_label is not None:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor="#cccccc", edgecolor="#2244cc",
                  linewidth=1.5, hatch="///", label="Outlier bin"),
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=8, framealpha=0.9, edgecolor="#aaaaaa")

    # ── State legend ──
    n_rollouts = len(all_rids)
    n_pass = sum(1 for r in all_rids if rewards.get(r, 0) >= 0.5)
    subtitle = f"{n_rollouts} rollouts, {n_pass} pass ({n_pass/n_rollouts:.0%}), {n_states} states, {len(all_steps)} steps"

    # ── Highlight selected rollout paths ──
    if highlight_rids:
        hl_colors = ["#e91e63", "#2196f3", "#ff9800", "#9c27b0"]
        for hi, hrid in enumerate(highlight_rids):
            color = hl_colors[hi % len(hl_colors)]
            pts_x, pts_y = [], []
            for step in all_steps:
                if (hrid, step) not in rid_step_to_state:
                    continue
                state = rid_step_to_state[(hrid, step)]
                if step not in step_bar_layout or state not in step_bar_layout[step]:
                    continue
                y_bot, h = step_bar_layout[step][state]
                rids_in = step_state_rids[step].get(state, [])
                if hrid in rids_in:
                    rank = sorted(rids_in).index(hrid)
                    y_pos = y_bot + h * (rank + 0.5) / max(len(rids_in), 1)
                else:
                    y_pos = y_bot + h / 2
                pts_x.append(x_centers[step])
                pts_y.append(y_pos)
            if pts_x:
                ax.plot(pts_x, pts_y, color=color, linewidth=2.5,
                        zorder=10, alpha=0.9, solid_capstyle="round")
                ax.scatter(pts_x, pts_y, color=color, s=50,
                           zorder=11, edgecolors="white", linewidths=1.0)
                outcome = "pass" if rewards.get(hrid, 0) >= 0.5 else "fail"
                ax.annotate(
                    f"r{hrid} ({outcome})", (pts_x[0], pts_y[0]),
                    textcoords="offset points", xytext=(-10, 12),
                    fontsize=8, fontweight="bold", color=color,
                    zorder=12,
                )

    x_max = max(x_centers.values()) + col_width
    ax.set_xlim(-col_width - 0.3, x_max + 2.0)
    ax.set_ylim(-1.5, total_height + 0.8)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

    plt.tight_layout()
    if return_fig:
        return fig
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_trajectory_flow_from_graph(
    G: nx.DiGraph,
    rollout_paths: dict[int, list[int]],
    indices: list[tuple[int, int]],
    labels: np.ndarray,
    rewards: dict[int, float],
    output_path: Path,
    title: str = "Trajectory Flow Through States",
):
    """Convenience wrapper: render flow diagram from already-built graph data."""
    render_trajectory_flow(labels, indices, rewards, output_path, title)


def _build_state_features(
    rollout_paths: dict[int, list[int]],
    all_rids: list[int],
    k: int,
) -> dict[str, np.ndarray]:
    """Build multiple feature representations from global state paths.

    Returns dict of feature_name -> X matrix.
    """
    n = len(all_rids)

    # 1. State visit fractions: what proportion of time in each state
    X_frac = np.zeros((n, k))
    for i, rid in enumerate(all_rids):
        path = rollout_paths[rid]
        for s in path:
            X_frac[i, s] += 1
        if len(path) > 0:
            X_frac[i] /= len(path)

    # 2. State visit binary: which states were visited
    X_visited = np.zeros((n, k))
    for i, rid in enumerate(all_rids):
        for s in set(rollout_paths[rid]):
            X_visited[i, s] = 1.0

    # 3. Transition bigram counts (normalized)
    X_bigram = np.zeros((n, k * k))
    for i, rid in enumerate(all_rids):
        path = rollout_paths[rid]
        for j in range(len(path) - 1):
            X_bigram[i, path[j] * k + path[j + 1]] += 1
        total = max(1, len(path) - 1)
        X_bigram[i] /= total

    # 4. First-N and last-N states (one-hot positional)
    window = min(5, min(len(rollout_paths[r]) for r in all_rids))
    X_first = np.zeros((n, window * k))
    X_last = np.zeros((n, window * k))
    for i, rid in enumerate(all_rids):
        path = rollout_paths[rid]
        for j in range(min(window, len(path))):
            X_first[i, j * k + path[j]] = 1.0
            X_last[i, j * k + path[-(j + 1)]] = 1.0

    # 5. Combined: fractions + bigrams + positional
    X_combined = np.hstack([X_frac, X_bigram, X_first, X_last])

    return {
        "visit_fractions": X_frac,
        "states_visited": X_visited,
        "bigrams": X_bigram,
        "positional": np.hstack([X_first, X_last]),
        "combined": X_combined,
    }


def global_path_cv_prediction(
    rollout_paths: dict[int, list[int]],
    outcome_map: dict[int, float],
    rewards: dict[int, float],
    k: int,
    label: str = "",
) -> dict:
    """Cross-validated R² prediction using global state path features."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    all_rids = sorted(set(rollout_paths.keys()) & set(outcome_map.keys()))
    y = np.array([outcome_map[r] for r in all_rids])

    if np.std(y) == 0:
        return {"error": "no variance in outcome"}

    n_folds, n_repeats = 5, 20

    def _cv_r2(X, y_arr):
        scores = []
        for _ in range(n_repeats):
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=np.random.randint(100000))
            for tr, te in kf.split(X):
                m = Ridge(alpha=1.0).fit(X[tr], y_arr[tr])
                yp = m.predict(X[te])
                ss_res = np.sum((y_arr[te] - yp) ** 2)
                ss_tot = np.sum((y_arr[te] - y_arr[te].mean()) ** 2)
                scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        return float(np.mean(scores))

    X_orig = np.array([[1 if rewards.get(r, 0) >= 0.5 else 0] for r in all_rids]).reshape(-1, 1)
    orig_cv = _cv_r2(X_orig, y)

    feature_sets = _build_state_features(rollout_paths, all_rids, k)
    feature_cv = {}
    for name, X in feature_sets.items():
        feature_cv[name] = _cv_r2(X, y)

    n_unique = len(set(tuple(rollout_paths[r]) for r in all_rids))
    n_singletons = sum(
        1 for _, cnt in Counter(tuple(rollout_paths[r]) for r in all_rids).items()
        if cnt == 1
    )

    print(f"\n{'='*60}")
    print(f"GLOBAL PATH CV R² — {label}")
    print(f"{'='*60}")
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


def plot_global_analysis(
    G: nx.DiGraph,
    rollout_paths: dict[int, list[int]],
    rewards: dict[int, float],
    output_path: Path,
    title_suffix: str = "",
):
    """Plot bottleneck detection and self-loop analysis for global graph."""
    special = {"START", "END_PASS", "END_FAIL"}
    regular_nodes = sorted(n for n in G.nodes if n not in special)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel 1: Node success rate vs visits (bubble plot)
    ax1 = axes[0, 0]
    for node in regular_nodes:
        d = G.nodes[node]
        color = _sr_color(d["success_rate"])
        ax1.scatter(d["visits"], d["success_rate"], s=d["n_rollouts"] * 10,
                    color=color, edgecolors="black", linewidth=0.5, alpha=0.8, zorder=5)
        ax1.annotate(f"S{node}", (d["visits"], d["success_rate"]),
                     fontsize=8, ha="center", textcoords="offset points", xytext=(0, 10))
    ax1.set_xlabel("Total Visits", fontsize=12)
    ax1.set_ylabel("Success Rate", fontsize=12)
    ax1.set_title(f"State Size vs Outcome{title_suffix}", fontsize=13)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax1.grid(alpha=0.3)

    # Panel 2: Self-loop analysis
    ax2 = axes[0, 1]
    self_loop_data = []
    for node in regular_nodes:
        if G.has_edge(node, node):
            sl_w = G.edges[node, node]["weight"]
            total_out = sum(G.edges[node, v]["weight"] for _, v in G.out_edges(node))
            sl_rate = sl_w / total_out if total_out > 0 else 0.0
            sl_sr = G.edges[node, node]["success_rate"]
            node_sr = G.nodes[node]["success_rate"]
            self_loop_data.append((node, sl_rate, sl_w, node_sr))
        else:
            total_out = sum(G.edges[node, v]["weight"] for _, v in G.out_edges(node))
            self_loop_data.append((node, 0.0, 0, G.nodes[node]["success_rate"]))

    if self_loop_data:
        nodes_sl = [f"S{d[0]}" for d in self_loop_data]
        rates = [d[1] for d in self_loop_data]
        counts = [d[2] for d in self_loop_data]
        srs = [d[3] for d in self_loop_data]
        colors = [_sr_color(sr) for sr in srs]
        x = np.arange(len(nodes_sl))
        bars = ax2.bar(x, rates, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(nodes_sl, fontsize=9)
        ax2.set_ylabel("Self-loop Rate", fontsize=12)
        ax2.set_title(f"Self-loop Rate by State{title_suffix}", fontsize=13)
        for i, (bar, c) in enumerate(zip(bars, counts)):
            if c > 0:
                ax2.text(i, bar.get_height() + 0.01, f"n={c}",
                         ha="center", fontsize=7)
        ax2.set_ylim(0, 1.05)
        ax2.grid(alpha=0.3, axis="y")

    # Panel 3: Outcome entropy per state
    ax3 = axes[1, 0]
    entropies = []
    for node in regular_nodes:
        sr = G.nodes[node]["success_rate"]
        ent = _binary_entropy(sr)
        entropies.append((node, ent, G.nodes[node]["visits"], sr))
    entropies.sort(key=lambda x: -x[1])
    if entropies:
        labels_e = [f"S{d[0]}" for d in entropies]
        ent_vals = [d[1] for d in entropies]
        visits_e = [d[2] for d in entropies]
        colors_e = [_sr_color(d[3]) for d in entropies]
        ax3.barh(range(len(labels_e)), ent_vals, color=colors_e, edgecolor="black", linewidth=0.5)
        ax3.set_yticks(range(len(labels_e)))
        ax3.set_yticklabels(labels_e, fontsize=9)
        for i, (e, v) in enumerate(zip(ent_vals, visits_e)):
            ax3.text(e + 0.01, i, f"({v} visits)", va="center", fontsize=7)
        ax3.set_xlabel("Outcome Entropy (bits)", fontsize=12)
        ax3.set_title(f"State Uncertainty{title_suffix}", fontsize=13)
        ax3.invert_yaxis()

    # Panel 4: Edge information gain (top transitions)
    ax4 = axes[1, 1]
    edge_ig = []
    node_ent_map = {n: _binary_entropy(G.nodes[n]["success_rate"]) for n in G.nodes if n not in special}
    for u, v, data in G.edges(data=True):
        if u in special or v in special:
            continue
        ig = node_ent_map.get(u, 0) - node_ent_map.get(v, 0)
        edge_ig.append((u, v, ig, data["weight"]))
    edge_ig.sort(key=lambda x: -abs(x[2]))

    if edge_ig:
        top = edge_ig[:15]
        labels_ig = [f"S{u}→S{v}" for u, v, _, _ in top]
        vals_ig = [ig for _, _, ig, _ in top]
        weights_ig = [w for _, _, _, w in top]
        colors_ig = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals_ig]
        ax4.barh(range(len(labels_ig)), vals_ig, color=colors_ig, edgecolor="white", height=0.7)
        ax4.set_yticks(range(len(labels_ig)))
        ax4.set_yticklabels(labels_ig, fontsize=9)
        for i, (_, w) in enumerate(zip(vals_ig, weights_ig)):
            ax4.text(0.001, i, f"  n={w}", va="center", fontsize=7)
        ax4.set_xlabel("Information Gain (bits)", fontsize=12)
        ax4.set_title(f"Edge Information Gain{title_suffix}", fontsize=13)
        ax4.axvline(0, color="black", lw=0.5)
        ax4.invert_yaxis()

    plt.suptitle(f"Global Thought-State Analysis{title_suffix}", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_global_cv_prediction(
    cv_results: dict[int, dict],
    output_path: Path,
    title_suffix: str = "",
):
    """Plot CV R² bar chart comparing feature types across K values."""
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
    k_labels = []
    for k in k_values:
        cv = cv_results[k]
        k_labels.append(f"K={k}\n({cv['n_unique_paths']} paths)")
    ax.set_xticklabels(k_labels, fontsize=9)
    ax.set_ylabel("Cross-validated R²", fontsize=12)
    ax.set_title(f"Global Thought-State Prediction by Feature Type{title_suffix}", fontsize=13)
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_ylim(-0.3, 1.05)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_global_analysis(
    npz_path: str,
    data_dir: str,
    task_id: str,
    out_dir: str,
    branch_dir: str | None = None,
    k_values: list[int] | None = None,
):
    """Full pipeline: global clustering -> graph -> analysis -> plots."""
    import glob

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    matrix, indices = _load_all_embeddings(npz_path)
    print(f"  {matrix.shape[0]} embeddings, dim={matrix.shape[1]}")

    rewards = load_rewards(data_dir, task_id)
    print(f"  {len(rewards)} rollout rewards loaded")

    branch_rates = None
    if branch_dir:
        branch_rates = {}
        for d in sorted(glob.glob(f"{branch_dir}/airline_12_*")):
            wi_id = d.split("/")[-1]
            parts = wi_id.split("_")
            step = int([p for p in parts if p.startswith("s")][0][1:])
            rid = int(parts[-1][1:])
            branches = sorted(glob.glob(f"{d}/branch_*.json"))
            if not branches:
                continue
            successes = sum(1 for bf in branches if json.load(open(bf)).get("is_success"))
            branch_rates[(step, rid)] = successes / len(branches)
        print(f"  {len(branch_rates)} branch rate entries loaded")

    outcome_map: dict[int, float]
    if branch_rates:
        by_rid: dict[int, list[float]] = defaultdict(list)
        for (step, rid), rate in branch_rates.items():
            by_rid[rid].append(rate)
        outcome_map = {rid: float(np.mean(rates)) for rid, rates in by_rid.items()}
        outcome_label = "branch SR"
    else:
        outcome_map = {r: float(rewards.get(r, 0)) for r in rewards}
        outcome_label = "reward"

    # Eigengap K for reference
    _, k_eigengap, eigvals = global_spectral_cluster(matrix, k=None)
    print(f"\nEigengap suggests K={k_eigengap}")

    if k_values is None:
        k_values = sorted(set([k_eigengap, 5, 8, 10, 15, 20]))

    all_cv_results: dict[int, dict] = {}
    best_k, best_cv = None, -999.0

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"K = {k}")
        print(f"{'='*60}")

        labels, actual_k, _ = global_spectral_cluster(matrix, k=k)
        state_counts = Counter(int(l) for l in labels)
        sizes = sorted(state_counts.values(), reverse=True)
        print(f"  State sizes: {sizes}")

        G, rollout_paths = build_global_thought_graph(labels, indices, rewards, branch_rates)
        n_self = sum(1 for u, v in G.edges() if u == v)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {n_self} self-loops")

        cv_result = global_path_cv_prediction(
            rollout_paths, outcome_map, rewards, k, f"K={k}, outcome={outcome_label}",
        )
        all_cv_results[k] = cv_result

        combined_cv = cv_result["feature_cv"].get("combined", -999)
        if combined_cv > best_cv:
            best_cv = combined_cv
            best_k = k

    print(f"\n{'='*60}")
    print(f"BEST K = {best_k} (combined CV-R² = {best_cv:.3f})")
    print(f"{'='*60}")

    # Produce final outputs for best K
    labels, _, _ = global_spectral_cluster(matrix, k=best_k)
    state_counts = Counter(int(l) for l in labels)

    metrics = {
        "k": best_k,
        "n_embeddings": len(labels),
        "assignments": [
            {"rollout_id": int(rid), "step": int(step), "state": int(labels[i])}
            for i, (rid, step) in enumerate(indices)
        ],
        "state_counts": {str(s): c for s, c in state_counts.items()},
        "all_k_cv_results": {
            str(k): {
                "orig_cv": v["orig_cv"],
                "feature_cv": v["feature_cv"],
                "n_unique_paths": v["n_unique_paths"],
                "n_singletons": v["n_singletons"],
            }
            for k, v in all_cv_results.items()
        },
    }
    with open(out / "global_clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    G, rollout_paths = build_global_thought_graph(labels, indices, rewards, branch_rates)
    n_self = sum(1 for u, v in G.edges() if u == v)
    print(f"\nFinal graph (K={best_k}): {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, {n_self} self-loops")

    print("Rendering graph...")
    render_global_graph(G, out / "global_thought_graph.png",
                        f"Global Thought States (K={best_k})")

    print("Rendering trajectory flow...")
    render_trajectory_flow(
        labels, indices, rewards,
        out / "trajectory_flow.png",
        f"Trajectory Flow (K={best_k})",
    )

    print("Running analysis plots...")
    plot_global_analysis(G, rollout_paths, rewards, out / "global_thought_analysis.png",
                         f" (K={best_k})")

    print("Rendering CV comparison...")
    plot_global_cv_prediction(all_cv_results, out / "global_thought_cv_prediction.png")

    print(f"\nAll outputs saved to: {out}")
    return G, rollout_paths, all_cv_results


# ══════════════════════════════════════════════════════════════════════
# Ablation study: predictive power vs rollout count
# ══════════════════════════════════════════════════════════════════════


def _fast_cv_r2(X: np.ndarray, y: np.ndarray, n_folds: int = 5, n_repeats: int = 10) -> float:
    """Lightweight cross-validated R² using Ridge regression."""
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


def _ablation_single_run(
    matrix: np.ndarray,
    indices: list[tuple[int, int]],
    sampled_rids: set[int],
    outcome_map: dict[int, float],
    rewards: dict[int, float],
    k: int,
) -> dict | None:
    """Run one (subsample, K) combination and return CV results."""
    mask = [i for i, (rid, _) in enumerate(indices) if rid in sampled_rids]
    if len(mask) < max(k + 1, 10):
        return None

    sub_matrix = matrix[mask]
    sub_indices = [indices[i] for i in mask]

    try:
        labels, actual_k, _ = global_spectral_cluster(sub_matrix, k=k)
    except Exception:
        return None

    if actual_k <= 1:
        return None

    rid_step_to_state = {}
    for (rid, step), state in zip(sub_indices, labels):
        rid_step_to_state[(rid, step)] = int(state)

    rollout_paths: dict[int, list[int]] = {}
    for rid in sorted(sampled_rids):
        steps_for_rid = sorted(s for r, s in sub_indices if r == rid)
        if not steps_for_rid:
            continue
        rollout_paths[rid] = [rid_step_to_state[(rid, s)] for s in steps_for_rid]

    all_rids = sorted(set(rollout_paths.keys()) & set(outcome_map.keys()))
    if len(all_rids) < 10:
        return None

    y = np.array([outcome_map[r] for r in all_rids])
    if np.std(y) == 0:
        return None

    X_orig = np.array([[1 if rewards.get(r, 0) >= 0.5 else 0] for r in all_rids]).reshape(-1, 1)
    orig_cv = _fast_cv_r2(X_orig, y)

    feature_sets = _build_state_features(rollout_paths, all_rids, k)
    feature_cv = {}
    for name, X in feature_sets.items():
        feature_cv[name] = _fast_cv_r2(X, y)

    n_unique = len(set(tuple(rollout_paths[r]) for r in all_rids))
    n_singletons = sum(
        1 for _, cnt in Counter(tuple(rollout_paths[r]) for r in all_rids).items()
        if cnt == 1
    )

    return {
        "orig_cv": orig_cv,
        "feature_cv": feature_cv,
        "n_rollouts": len(all_rids),
        "n_unique_paths": n_unique,
        "n_singletons": n_singletons,
        "k": k,
    }


def run_ablation_study(
    npz_path: str,
    data_dir: str,
    task_id: str,
    branch_dir: str,
    out_dir: str,
    sizes: list[int] | None = None,
    k_values: list[int] | None = None,
    n_seeds: int = 20,
):
    """Ablation: how does predictive power scale with rollout count?"""
    import glob
    import time

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    matrix, indices = _load_all_embeddings(npz_path)
    rewards = load_rewards(data_dir, task_id)

    branch_rates: dict[tuple[int, int], float] = {}
    for d in sorted(glob.glob(f"{branch_dir}/airline_12_*")):
        wi_id = d.split("/")[-1]
        parts = wi_id.split("_")
        step = int([p for p in parts if p.startswith("s")][0][1:])
        rid = int(parts[-1][1:])
        branches = sorted(glob.glob(f"{d}/branch_*.json"))
        if not branches:
            continue
        successes = sum(1 for bf in branches if json.load(open(bf)).get("is_success"))
        branch_rates[(step, rid)] = successes / len(branches)

    by_rid: dict[int, list[float]] = defaultdict(list)
    for (step, rid), rate in branch_rates.items():
        by_rid[rid].append(rate)
    outcome_map = {rid: float(np.mean(rates)) for rid, rates in by_rid.items()}

    all_rids_with_branches = sorted(outcome_map.keys())
    max_n = len(all_rids_with_branches)
    print(f"  {matrix.shape[0]} embeddings, {len(rewards)} rewards, "
          f"{max_n} rollouts with branch data")

    if sizes is None:
        sizes = [10, 15, 20, 30, 40, 50, 60, 70, 80, max_n]
    sizes = [s for s in sizes if s <= max_n]

    if k_values is None:
        k_values = [3, 5, 8, 10, 15, 20]

    total_runs = len(sizes) * n_seeds * len(k_values)
    print(f"\nAblation: {len(sizes)} sizes x {n_seeds} seeds x {len(k_values)} K = {total_runs} runs")

    results: list[dict] = []
    t0 = time.time()
    done = 0

    for size in sizes:
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed * 7919 + size)
            if size >= max_n:
                sampled = set(all_rids_with_branches)
            else:
                sampled = set(rng.choice(all_rids_with_branches, size=size, replace=False).tolist())

            for k in k_values:
                if k >= len(sampled):
                    continue
                res = _ablation_single_run(matrix, indices, sampled, outcome_map, rewards, k)
                done += 1
                if res is not None:
                    res["size"] = size
                    res["seed"] = seed
                    results.append(res)

            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_runs - done) / rate if rate > 0 else 0
                print(f"  {done}/{total_runs} runs ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    print(f"\nCompleted {len(results)} successful runs in {elapsed:.1f}s")

    with open(out / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved raw results to {out / 'ablation_results.json'}")

    _plot_ablation(results, sizes, out)
    return results


def _plot_ablation(results: list[dict], sizes: list[int], out: Path):
    """Generate the three ablation plots."""
    from scipy.optimize import curve_fit

    # ── Organize results by (size, seed) → best K result ──
    by_size_seed: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in results:
        by_size_seed[(r["size"], r["seed"])].append(r)

    # For each (size, seed), pick the K with best combined CV-R²
    best_by_size_seed: dict[tuple[int, int], dict] = {}
    for (size, seed), runs in by_size_seed.items():
        best = max(runs, key=lambda r: r["feature_cv"].get("combined", -999))
        best_by_size_seed[(size, seed)] = best

    by_size: dict[int, list[dict]] = defaultdict(list)
    for (size, _), best in best_by_size_seed.items():
        by_size[size].append(best)

    sorted_sizes = sorted(by_size.keys())

    # ── Plot A: Predictive power vs rollout count ──
    fig, ax = plt.subplots(figsize=(12, 7))

    medians, q25s, q75s, means = [], [], [], []
    orig_medians = []
    for size in sorted_sizes:
        vals = [r["feature_cv"]["combined"] for r in by_size[size]]
        orig_vals = [r["orig_cv"] for r in by_size[size]]
        medians.append(np.median(vals))
        q25s.append(np.percentile(vals, 25))
        q75s.append(np.percentile(vals, 75))
        means.append(np.mean(vals))
        orig_medians.append(np.median(orig_vals))

    x = np.array(sorted_sizes)
    med = np.array(medians)
    q25 = np.array(q25s)
    q75 = np.array(q75s)

    ax.fill_between(x, q25, q75, alpha=0.2, color="#2ecc71", label="IQR (combined)")
    ax.plot(x, med, "s-", color="#2ecc71", lw=2, ms=8, label="Median combined CV-R²", zorder=5)
    ax.plot(x, orig_medians, "o--", color="#9b59b6", lw=2, ms=6,
            label="Median orig outcome CV-R²")

    # Fit log trendline: R² = a * log(N) + b
    try:
        valid = [(s, m) for s, m in zip(sorted_sizes, medians) if m > 0]
        if len(valid) >= 3:
            xs_fit = np.array([v[0] for v in valid])
            ys_fit = np.array([v[1] for v in valid])

            def _log_model(n, a, b):
                return a * np.log(n) + b

            popt, _ = curve_fit(_log_model, xs_fit, ys_fit, p0=[0.2, -0.5])
            x_extrap = np.array([*sorted_sizes, 100, 150, 200, 300, 500])
            x_extrap = np.sort(np.unique(x_extrap))
            y_extrap = _log_model(x_extrap, *popt)
            y_extrap = np.clip(y_extrap, -0.5, 1.0)
            ax.plot(x_extrap, y_extrap, ":", color="#2ecc71", lw=1.5, alpha=0.6,
                    label=f"Log fit: {popt[0]:.3f}·ln(N){popt[1]:+.3f}")

            for n_ext in [100, 200, 500]:
                y_ext = _log_model(n_ext, *popt)
                ax.scatter(n_ext, y_ext, marker="*", s=150, color="#e74c3c", zorder=10)
                ax.annotate(f"N={n_ext}: {y_ext:.3f}", (n_ext, y_ext),
                            fontsize=8, textcoords="offset points", xytext=(5, 10))
    except Exception:
        pass

    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Cross-validated R² (branch SR)", fontsize=12)
    ax.set_title("Predictive Power vs Rollout Count\n(best K per subsample, 20 seeds)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.3, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", ls="-", lw=0.5)

    plt.tight_layout()
    fig.savefig(out / "ablation_power_vs_n.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / 'ablation_power_vs_n.png'}")

    # ── Plot B: Optimal K vs rollout count ──
    fig, ax = plt.subplots(figsize=(12, 7))

    k_by_size = defaultdict(list)
    for (size, _), best in best_by_size_seed.items():
        k_by_size[size].append(best["k"])

    k_medians, k_q25, k_q75 = [], [], []
    for size in sorted_sizes:
        vals = k_by_size[size]
        k_medians.append(np.median(vals))
        k_q25.append(np.percentile(vals, 25))
        k_q75.append(np.percentile(vals, 75))

    ax.fill_between(x, k_q25, k_q75, alpha=0.2, color="#3498db")
    ax.plot(x, k_medians, "D-", color="#3498db", lw=2, ms=8, label="Median best K")

    # Scatter all K values for transparency
    for size in sorted_sizes:
        vals = k_by_size[size]
        jitter = np.random.RandomState(42).uniform(-0.5, 0.5, len(vals))
        ax.scatter([size] * len(vals), vals, alpha=0.2, s=20, color="#3498db")

    # Fit sqrt heuristic: K ~ a * sqrt(N) + b
    try:
        xs_k = np.array(sorted_sizes)
        ys_k = np.array(k_medians)

        def _sqrt_model(n, a, b):
            return a * np.sqrt(n) + b

        popt_k, _ = curve_fit(_sqrt_model, xs_k, ys_k, p0=[1.0, 0.0])
        x_k_ext = np.linspace(min(sorted_sizes), 500, 100)
        y_k_ext = _sqrt_model(x_k_ext, *popt_k)
        ax.plot(x_k_ext, y_k_ext, ":", color="#e74c3c", lw=1.5,
                label=f"Fit: {popt_k[0]:.2f}·√N{popt_k[1]:+.1f}")
    except Exception:
        pass

    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Best K (number of global states)", fontsize=12)
    ax.set_title("Optimal K vs Rollout Count\n(K with highest combined CV-R²)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "ablation_best_k_vs_n.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / 'ablation_best_k_vs_n.png'}")

    # ── Plot C: Feature type comparison across sizes ──
    fig, ax = plt.subplots(figsize=(12, 7))

    feature_names = ["combined", "states_visited", "bigrams", "positional", "visit_fractions"]
    colors_map = {
        "combined": "#e74c3c",
        "states_visited": "#e67e22",
        "bigrams": "#2ecc71",
        "positional": "#9b59b6",
        "visit_fractions": "#3498db",
    }

    for fname in feature_names:
        feat_medians = []
        feat_q25 = []
        feat_q75 = []
        for size in sorted_sizes:
            vals = [r["feature_cv"].get(fname, 0) for r in by_size[size]]
            feat_medians.append(np.median(vals))
            feat_q25.append(np.percentile(vals, 25))
            feat_q75.append(np.percentile(vals, 75))
        color = colors_map.get(fname, "gray")
        ax.fill_between(x, feat_q25, feat_q75, alpha=0.1, color=color)
        ax.plot(x, feat_medians, "o-", color=color, lw=2, ms=5, label=fname)

    ax.plot(x, orig_medians, "k--", lw=2, label="original outcome")
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.set_xlabel("Number of Rollouts", fontsize=12)
    ax.set_ylabel("Cross-validated R² (branch SR)", fontsize=12)
    ax.set_title("Feature Type Prediction vs Rollout Count\n(at best K per subsample)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.5, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "ablation_features_vs_n.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / 'ablation_features_vs_n.png'}")


# ══════════════════════════════════════════════════════════════════════
# Multi-task analysis: discover tasks, run global analysis, compare
# ══════════════════════════════════════════════════════════════════════


def _tdp_sort_key(filepath: str) -> int:
    """Extract numeric rollout index from a TDP filename for correct sorting."""
    import re as _re
    m = _re.search(r"_tdp_(\d+)\.json$", filepath)
    return int(m.group(1)) if m else 0


def _discover_tasks(data_dir: str, embed_dir: str) -> list[dict]:
    """Discover all tasks from NPZ files and map back to TDP files.

    Returns list of dicts with keys: task_label, npz_path, task_id, domain, short_id
    """
    import re as _re

    npz_dir = Path(embed_dir)
    results = []

    for npz_file in sorted(npz_dir.glob("*.npz")):
        stem = npz_file.stem  # e.g. "airline_12" or "telecom_[mobile_data...]"

        # Determine domain from prefix
        domain = stem.split("_")[0]  # airline, retail, telecom
        domain_dir = Path(data_dir) / domain

        if not domain_dir.exists():
            continue

        # Find matching TDP prefix by checking which files exist
        # For simple tasks (airline_12 -> 12_tdp_*), just strip domain prefix
        rest = stem[len(domain) + 1:]  # everything after "domain_"

        # Try simple numeric ID first
        if _re.match(r"^\d+$", rest):
            task_id = f"{domain}/{rest}"
            short_id = f"{domain}/{rest}"
            # Try tdp format first: {id}_tdp_*.json
            test_files = list(domain_dir.glob(f"{rest}_tdp_*.json"))
            if test_files:
                results.append({
                    "task_label": short_id,
                    "npz_path": str(npz_file),
                    "task_id": task_id,
                    "domain": domain,
                    "short_id": short_id,
                    "tdp_files": sorted((str(f) for f in test_files), key=_tdp_sort_key),
                })
                continue
            # Fallback: consolidated format {id}/rollout_*.json
            task_subdir = domain_dir / rest
            if task_subdir.is_dir():
                rollout_files = sorted(task_subdir.glob("rollout_*.json"))
                if rollout_files:
                    results.append({
                        "task_label": short_id,
                        "npz_path": str(npz_file),
                        "task_id": task_id,
                        "domain": domain,
                        "short_id": short_id,
                        "tdp_files": [str(f) for f in rollout_files],
                    })
                    continue

        # For complex task IDs (telecom), scan TDP files to match
        # The NPZ stem has sanitized chars; we need to find the original
        all_tdp = sorted(domain_dir.glob("*_tdp_*.json"))
        prefix_groups: dict[str, list[str]] = {}
        for f in all_tdp:
            fname = f.name
            m = _re.match(r"^(.+)_tdp_\d+\.json$", fname)
            if m:
                prefix_groups.setdefault(m.group(1), []).append(str(f))

        # Match by sanitizing each prefix the same way embed_openai does
        # task_id_to_filename: sanitize then truncate+hash if >200 chars
        import hashlib as _hashlib

        matched = False
        for prefix, files in prefix_groups.items():
            sanitized = prefix.replace("|", "_").replace(":", "_")
            candidate_stem = f"{domain}_{sanitized}"
            # Also try the truncated+hashed form for very long names
            if len(candidate_stem) > 200:
                h = _hashlib.sha256(f"{domain}/{prefix}".encode()).hexdigest()[:8]
                candidate_stem_hashed = candidate_stem[:190] + f"__{h}"
            else:
                candidate_stem_hashed = None

            if candidate_stem == stem or (candidate_stem_hashed and candidate_stem_hashed == stem):
                task_id = f"{domain}/{prefix}"
                short_label = prefix
                if len(short_label) > 40:
                    h = _hashlib.sha256(prefix.encode()).hexdigest()[:6]
                    short_label = short_label[:33] + f"_{h}"
                short_id = f"{domain}/{short_label}"
                results.append({
                    "task_label": short_id,
                    "npz_path": str(npz_file),
                    "task_id": task_id,
                    "domain": domain,
                    "short_id": short_id,
                    "tdp_files": sorted(files, key=_tdp_sort_key),
                })
                matched = True
                break

        if not matched:
            print(f"  WARNING: Could not map NPZ {npz_file.name} to TDP files")

    return results


def _load_rewards_from_files(tdp_files: list[str]) -> dict[int, float]:
    """Load rewards directly from a list of TDP/rollout file paths.

    Keys are **enumeration indices** (position in the sorted file list),
    matching the ``r_idx`` convention used by the embedding NPZ files.

    Supports both formats:
      - {id}_tdp_{num}.json (legacy)
      - rollout_{num}.json (consolidated)
    """
    import re as _re
    import warnings

    rewards: dict[int, float] = {}
    for enum_idx, fp in enumerate(tdp_files):
        fname = Path(fp).name
        # Try tdp format
        m = _re.search(r"_tdp_(\d+)\.json$", fname)
        if m:
            file_num = int(m.group(1))
            if file_num != enum_idx:
                warnings.warn(
                    f"TDP file numbering gap: enum {enum_idx} -> {fname}",
                    stacklevel=2,
                )
            with open(fp) as fh:
                rewards[enum_idx] = json.load(fh).get("reward", 0.0)
            continue
        # Try consolidated rollout format
        m = _re.search(r"rollout_(\d+)\.json$", fname)
        if m:
            with open(fp) as fh:
                rewards[enum_idx] = json.load(fh).get("reward", 0.0)
            continue
    return rewards


def run_multi_task_analysis(
    data_dir: str,
    embed_dir: str,
    out_dir: str,
    branch_dir: str | None = None,
    k_values: list[int] | None = None,
):
    """Run global thought-state analysis on all tasks and produce cross-task comparison."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Discovering tasks...")
    tasks = _discover_tasks(data_dir, embed_dir)
    print(f"  Found {len(tasks)} tasks:")
    for t in tasks:
        print(f"    {t['short_id']} — {len(t['tdp_files'])} TDP files, NPZ={Path(t['npz_path']).name}")

    if k_values is None:
        k_values = [3, 5, 8, 10, 15, 20]

    all_task_results: dict[str, dict] = {}

    for ti, task in enumerate(tasks):
        label = task["short_id"]
        task_out = out / task["task_label"].replace("/", "_")
        task_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# Task {ti+1}/{len(tasks)}: {label}")
        print(f"{'#'*70}")

        matrix, indices = _load_all_embeddings(task["npz_path"])
        rewards = _load_rewards_from_files(task["tdp_files"])
        n_pass = sum(1 for r in rewards.values() if r >= 0.5)
        sr = n_pass / len(rewards) if rewards else 0
        print(f"  {matrix.shape[0]} embeddings, {len(rewards)} rollouts, SR={sr:.0%}")

        # For airline/12 with branch data, load branches
        branch_rates = None
        if branch_dir and task["task_id"] == "airline/12":
            import glob as _glob
            branch_rates = {}
            for d in sorted(_glob.glob(f"{branch_dir}/airline_12_*")):
                wi_id = d.split("/")[-1]
                parts = wi_id.split("_")
                step = int([p for p in parts if p.startswith("s")][0][1:])
                rid = int(parts[-1][1:])
                branches = sorted(_glob.glob(f"{d}/branch_*.json"))
                if not branches:
                    continue
                successes = sum(1 for bf in branches if json.load(open(bf)).get("is_success"))
                branch_rates[(step, rid)] = successes / len(branches)
            print(f"  {len(branch_rates)} branch rate entries loaded")

        # Determine outcome map
        if branch_rates:
            by_rid: dict[int, list[float]] = defaultdict(list)
            for (step, rid), rate in branch_rates.items():
                by_rid[rid].append(rate)
            outcome_map = {rid: float(np.mean(rates)) for rid, rates in by_rid.items()}
            outcome_label = "branch SR"
        else:
            outcome_map = {r: float(rewards.get(r, 0)) for r in rewards}
            outcome_label = "reward"

        # Eigengap
        _, k_eigengap, _ = global_spectral_cluster(matrix, k=None)
        print(f"  Eigengap K = {k_eigengap}")

        task_k_values = sorted(set([k_eigengap] + k_values))

        all_cv: dict[int, dict] = {}
        best_k, best_cv = None, -999.0

        for k in task_k_values:
            if k < 2 or k >= matrix.shape[0]:
                continue
            labels, actual_k, _ = global_spectral_cluster(matrix, k=k)
            G, rollout_paths = build_global_thought_graph(labels, indices, rewards, branch_rates)
            cv_result = global_path_cv_prediction(
                rollout_paths, outcome_map, rewards, k, f"{label} K={k}",
            )
            all_cv[k] = cv_result
            combined_cv = cv_result["feature_cv"].get("combined", -999)
            if combined_cv > best_cv:
                best_cv = combined_cv
                best_k = k

        print(f"\n  BEST K = {best_k} (combined CV-R² = {best_cv:.3f})")

        # Final outputs for best K
        labels, _, _ = global_spectral_cluster(matrix, k=best_k)
        G, rollout_paths = build_global_thought_graph(labels, indices, rewards, branch_rates)

        render_global_graph(G, task_out / "global_thought_graph.png",
                            f"Global Thought States — {label} (K={best_k})")
        render_trajectory_flow(
            labels, indices, rewards,
            task_out / "trajectory_flow.png",
            f"Trajectory Flow — {label} (K={best_k})",
        )
        plot_global_analysis(G, rollout_paths, rewards,
                             task_out / "global_thought_analysis.png", f" {label} K={best_k}")
        plot_global_cv_prediction(all_cv, task_out / "global_thought_cv_prediction.png",
                                  f"\n{label}")

        # Save metrics
        state_counts = Counter(int(l) for l in labels)
        metrics = {
            "task_id": task["task_id"],
            "task_label": label,
            "domain": task["domain"],
            "n_rollouts": len(rewards),
            "success_rate": sr,
            "outcome_label": outcome_label,
            "n_embeddings": len(labels),
            "eigengap_k": k_eigengap,
            "best_k": best_k,
            "best_combined_cv_r2": best_cv,
            "state_counts": {str(s): c for s, c in state_counts.items()},
            "all_k_cv_results": {
                str(kv): {
                    "orig_cv": v["orig_cv"],
                    "feature_cv": v["feature_cv"],
                    "n_unique_paths": v["n_unique_paths"],
                    "n_singletons": v["n_singletons"],
                }
                for kv, v in all_cv.items()
            },
        }
        with open(task_out / "global_clustering_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        all_task_results[label] = metrics
        print(f"  Saved outputs to {task_out}")

    # Cross-task comparison
    print(f"\n{'='*70}")
    print("CROSS-TASK COMPARISON")
    print(f"{'='*70}")

    with open(out / "all_tasks_summary.json", "w") as f:
        json.dump(all_task_results, f, indent=2)

    _plot_cross_task_comparison(all_task_results, out)

    return all_task_results


def _plot_cross_task_comparison(results: dict[str, dict], out: Path):
    """Generate cross-task comparison plots."""
    tasks = sorted(results.keys())
    n = len(tasks)

    # Shorten labels for display
    short_labels = []
    for t in tasks:
        parts = t.split("/")
        domain = parts[0][:3].upper()
        tid = parts[1] if len(parts) > 1 else t
        if len(tid) > 15:
            tid = tid[:12] + "..."
        short_labels.append(f"{domain}/{tid}")

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # ── Panel 1: Best combined CV-R² per task ──
    ax1 = axes[0, 0]
    best_cvs = [results[t]["best_combined_cv_r2"] for t in tasks]
    srs = [results[t]["success_rate"] for t in tasks]
    domain_colors = {"airline": "#3498db", "retail": "#e67e22", "telecom": "#2ecc71"}
    colors = [domain_colors.get(results[t]["domain"], "gray") for t in tasks]

    bars = ax1.bar(range(n), best_cvs, color=colors, edgecolor="white", width=0.7)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Best Combined CV-R²", fontsize=11)
    ax1.set_title("Predictive Power per Task (Best K)", fontsize=13)
    ax1.axhline(0, color="gray", ls="-", lw=0.5)
    ax1.grid(alpha=0.3, axis="y")
    for i, (cv, sr) in enumerate(zip(best_cvs, srs)):
        ax1.text(i, cv + 0.02, f"K={results[tasks[i]]['best_k']}\nSR={sr:.0%}",
                 ha="center", va="bottom", fontsize=7)

    # Legend for domains
    for dom, col in domain_colors.items():
        ax1.bar([], [], color=col, label=dom)
    ax1.legend(fontsize=9)

    # ── Panel 2: Feature type comparison across tasks (at best K) ──
    ax2 = axes[0, 1]
    feature_names = ["combined", "bigrams", "visit_fractions", "states_visited", "positional"]
    feat_colors = {
        "combined": "#e74c3c", "bigrams": "#2ecc71",
        "visit_fractions": "#3498db", "states_visited": "#e67e22",
        "positional": "#9b59b6",
    }
    width = 0.15
    x = np.arange(n)
    for j, fname in enumerate(feature_names):
        vals = []
        for t in tasks:
            best_k = str(results[t]["best_k"])
            cv_data = results[t]["all_k_cv_results"].get(best_k, {})
            vals.append(cv_data.get("feature_cv", {}).get(fname, 0))
        ax2.bar(x + j * width, vals, width, label=fname,
                color=feat_colors.get(fname, f"C{j}"), edgecolor="white")

    # Add original outcome baseline
    orig_vals = []
    for t in tasks:
        best_k = str(results[t]["best_k"])
        orig_vals.append(results[t]["all_k_cv_results"].get(best_k, {}).get("orig_cv", 0))
    ax2.bar(x + len(feature_names) * width, orig_vals, width, label="orig outcome",
            color="#555", edgecolor="white")

    ax2.set_xticks(x + (len(feature_names) / 2) * width)
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("CV-R²", fontsize=11)
    ax2.set_title("Feature Type Comparison (at Best K)", fontsize=13)
    ax2.axhline(0, color="gray", ls="-", lw=0.5)
    ax2.legend(fontsize=7, ncol=3, loc="upper right")
    ax2.grid(alpha=0.3, axis="y")

    # ── Panel 3: Success rate vs best CV-R² scatter ──
    ax3 = axes[1, 0]
    for i, t in enumerate(tasks):
        ax3.scatter(srs[i], best_cvs[i], s=120, c=colors[i], edgecolors="black",
                    linewidths=0.5, zorder=5)
        ax3.annotate(short_labels[i], (srs[i], best_cvs[i]),
                     fontsize=7, textcoords="offset points", xytext=(5, 5))
    ax3.set_xlabel("Task Success Rate", fontsize=11)
    ax3.set_ylabel("Best Combined CV-R²", fontsize=11)
    ax3.set_title("Success Rate vs Predictive Power", fontsize=13)
    ax3.grid(alpha=0.3)
    ax3.axhline(0, color="gray", ls="-", lw=0.5)

    # ── Panel 4: K sweep comparison across tasks (heatmap) ──
    ax4 = axes[1, 1]
    all_ks = sorted(set(
        int(k) for t in tasks for k in results[t]["all_k_cv_results"].keys()
    ))
    heatmap_data = np.full((n, len(all_ks)), np.nan)
    for i, t in enumerate(tasks):
        for j, k in enumerate(all_ks):
            cv_data = results[t]["all_k_cv_results"].get(str(k), {})
            heatmap_data[i, j] = cv_data.get("feature_cv", {}).get("combined", np.nan)

    im = ax4.imshow(heatmap_data, aspect="auto", cmap="RdYlGn",
                    vmin=-0.3, vmax=max(0.5, np.nanmax(heatmap_data)))
    ax4.set_xticks(range(len(all_ks)))
    ax4.set_xticklabels([str(k) for k in all_ks], fontsize=9)
    ax4.set_yticks(range(n))
    ax4.set_yticklabels(short_labels, fontsize=9)
    ax4.set_xlabel("K (number of clusters)", fontsize=11)
    ax4.set_title("Combined CV-R² Heatmap (task × K)", fontsize=13)
    plt.colorbar(im, ax=ax4, label="CV-R²", shrink=0.8)

    # Annotate best K per task
    for i, t in enumerate(tasks):
        best_k = results[t]["best_k"]
        if best_k in all_ks:
            j = all_ks.index(best_k)
            ax4.scatter(j, i, marker="*", s=200, c="black", zorder=10)

    plt.suptitle("Cross-Task Global Thought-State Analysis", fontsize=16,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out / "cross_task_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / 'cross_task_comparison.png'}")

    # ── Summary table ──
    print(f"\n{'Task':<45s} {'Domain':<10s} {'SR':>5s} {'K':>3s} {'CV-R²':>7s} {'Eigengap':>8s}")
    print("-" * 85)
    for t in tasks:
        r = results[t]
        print(f"{t:<45s} {r['domain']:<10s} {r['success_rate']:>5.0%} "
              f"{r['best_k']:>3d} {r['best_combined_cv_r2']:>7.3f} {r['eigengap_k']:>8d}")


if __name__ == "__main__":
    main()
