#!/usr/bin/env python3
"""Build the per-task JSON bundles behind the project page's interactive figures.

For one (benchmark, policy, task) cell the bundle holds everything the browser
needs and nothing it does not: the step texts, the Fig. 1 behavior-state
clustering (HDBSCAN, mcs=5, outlier bin; Appendix A), and the Precog encoder's
128-d latent vector for every rollout prefix, so prototypes, scores, AUROC and
the 2-D map can all be recomputed live for any support set. The 3072-d step
embeddings never leave this machine.

Reads the release layout under $PRECOG_DATA (scripts/download.py):
    rollouts/<benchmark>/<policy>/<task>/rollout_NNN.json
    embeddings/<benchmark>/<policy>/<task>/{action,observation}.npz
    ckpts/prefix/prefix_<benchmark>_g-q-d_uniform_s0.pt   (prefix-trained encoder, Sec. 4.5 / Sec. 5)
    ckpts/precog/precog_agentclinic-bfclv3-tau2_all_s0.pt (generalist full-rollout encoder, Table 2)

    python scripts/build_web_bundles.py --benchmark tau2 --policy qwen3-80b --task retail_103
    python scripts/build_web_bundles.py --benchmark tau2 --policy qwen3-80b            # every task in the cell
Writes docs/data/<benchmark>/<policy>/<task>.json (+ docs/data/index.json).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA, REPO, CKPTS, POLICY_DIR, DATASETS  # noqa: E402

OUT = REPO / "docs" / "data"
MCS, NOISE = 5, "outlier_bin"          # Fig. 1 / Appendix A protocol
MAX_TEXT = 4000                        # chars kept per action / observation
ROUND = 4
STOP_THRESHOLD = 0.8                   # EarlyStopConfig.threshold (official runs)
MODELS = {
    "prefix": lambda b: CKPTS / "prefix" / f"prefix_{b}_g-q-d_uniform_s0.pt",
    "full": lambda b: CKPTS / "precog" / "precog_agentclinic-bfclv3-tau2_all_s0.pt",
}


def load_task(benchmark, policy, task):
    root = DATA / "rollouts" / benchmark / policy / task
    emb = DATA / "embeddings" / benchmark / policy / task
    info = json.load(open(root / "task.json"))
    act = np.load(emb / "action.npz")
    obs = np.load(emb / "observation.npz")
    rewards = {int(k): float(v) for k, v in json.load(open(emb / "rewards.json")).items()}
    rollouts = []
    for f in sorted(root.glob("rollout_*.json")):
        d = json.load(open(f))
        rid, T = int(d["rid"]), int(d["n_steps"])
        keys = [f"r{rid}_s{t}" for t in range(T)]
        if not all(k in act.files for k in keys):
            print(f"  skip rid {rid}: embeddings missing"); continue
        a = np.stack([act[k].reshape(-1) for k in keys]).astype(np.float32)
        o = np.stack([obs[k].reshape(-1) for k in keys]).astype(np.float32)
        steps = [{"a": s["action"][:MAX_TEXT], "o": s["observation"][:MAX_TEXT]} for s in d["steps"]]
        rollouts.append({"rid": rid, "reward": float(rewards.get(rid, d["reward"])), "steps": steps, "act": a, "obs": o})
    rollouts.sort(key=lambda r: r["rid"])
    return info, rollouts


def cluster_steps(rollouts):
    """Fig. 1 clustering: HDBSCAN over 0.5*(cos_act + cos_obs), outliers -> one extra bin."""
    from src.prm.paired_action_response import hdbscan_cluster_affinity, paired_similarity
    A = np.concatenate([r["act"] for r in rollouts]); O = np.concatenate([r["obs"] for r in rollouts])
    S, _, _ = paired_similarity(A, O, "mean")
    labels = hdbscan_cluster_affinity(S, min_cluster_size=MCS, noise_strategy=NOISE)
    K = int(labels.max()) + 1
    # with outlier_bin the outlier bin is the last label iff any point was noise; detect via HDBSCAN rerun
    from sklearn.cluster import HDBSCAN
    dist = 1.0 - np.clip(S, 0, 1); np.fill_diagonal(dist, 0.0)
    raw = HDBSCAN(min_cluster_size=MCS, metric="precomputed").fit_predict(dist)
    noise_label = K - 1 if (raw == -1).any() else None
    # per-cluster stats + medoid (highest mean similarity to its members)
    i = 0
    owner = []
    for r in rollouts:
        for t in range(len(r["steps"])):
            owner.append((r["rid"], t, r["reward"] >= 0.5)); i += 1
    clusters = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        members = [owner[j] for j in idx]
        sub = S[np.ix_(idx, idx)]
        med = int(idx[np.argmax(sub.mean(axis=1))])
        clusters.append({"id": k, "n": int(len(idx)), "n_rollouts": len({m[0] for m in members}),
                         "sr": round(float(np.mean([m[2] for m in members])), ROUND),
                         "noise": k == noise_label, "medoid": [owner[med][0], owner[med][1]]})
    # write labels back
    i = 0
    for r in rollouts:
        for t in range(len(r["steps"])):
            r["steps"][t]["c"] = int(labels[i]); i += 1
    return clusters


def encode_prefixes(model, rollouts, all_prefixes):
    """128-d latent for every prefix (t = 1..T) of every rollout, or just the full rollout."""
    import torch
    out = []
    for r in rollouts:
        T = len(r["steps"])
        ts = list(range(1, T + 1)) if all_prefixes else [T]
        toks, poss = zip(*[model._interleave_rollout(r["act"][:t], r["obs"][:t]) for t in ts])
        with torch.no_grad():
            z = model._encode_batch(list(toks), list(poss)).cpu().numpy()
        out.append(np.round(z, ROUND).tolist())
    return out


def pca2(Z):
    Z = np.asarray(Z, dtype=np.float64)
    mean = Z.mean(axis=0)
    U, s, Vt = np.linalg.svd(Z - mean, full_matrices=False)
    var = (s ** 2) / max(1, len(Z) - 1)
    return {"mean": np.round(mean, ROUND).tolist(), "components": np.round(Vt[:2], ROUND).tolist(),
            "explained": np.round(var[:2] / var.sum(), ROUND).tolist()}


def cv_folds(rollouts, n=3, seed=0):
    from sklearn.model_selection import StratifiedKFold
    y = np.array([int(r["reward"] >= 0.5) for r in rollouts])
    rids = np.array([r["rid"] for r in rollouts])
    skf = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    return [sorted(int(x) for x in rids[test]) for _, test in skf.split(rids, y)]


def task_label(benchmark, task):
    if benchmark == "tau2":
        dom, idx = task.split("_", 1); return f"{dom}/{idx}"
    if benchmark == "bfclv3":
        return task.replace("_", "/", 1)
    return f"medqa/{int(task.split('_')[1])}"


def build(benchmark, policy, task, models, held_out):
    info, rollouts = load_task(benchmark, policy, task)
    print(f"[{benchmark}/{policy}/{task}] {len(rollouts)} rollouts, {sum(len(r['steps']) for r in rollouts)} steps", flush=True)
    clusters = cluster_steps(rollouts)
    enc = {}
    for name, m in models.items():
        z = encode_prefixes(m, rollouts, all_prefixes=(name == "prefix"))
        finals = [zz[-1] for zz in z]
        enc[name] = {"checkpoint": Path(MODELS[name](benchmark)).name, "platt": [m.platt_a, m.platt_b],
                     "temperature": m.temperature, "distance": getattr(m, "distance", "euclid"),
                     "prefix": name == "prefix", "pca": pca2(finals), "z": z}
    bundle = {
        "benchmark": benchmark, "policy": policy, "task": task, "task_id": task_label(benchmark, task),
        "domain": info.get("domain") or info.get("category"), "success_rate": info.get("success_rate"),
        "held_out": task_label(benchmark, task) in held_out, "n_rollouts": len(rollouts),
        "cluster_protocol": {"method": "hdbscan", "min_cluster_size": MCS, "affinity": "0.5*(cos_action+cos_observation)", "noise": NOISE},
        "stop_threshold": STOP_THRESHOLD, "cv_folds": cv_folds(rollouts),
        "clusters": clusters,
        "rollouts": [{"rid": r["rid"], "reward": r["reward"], "n_steps": len(r["steps"]), "steps": r["steps"]} for r in rollouts],
        "encoders": enc,
    }
    out = OUT / benchmark / policy / f"{task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(bundle, open(out, "w"), separators=(",", ":"))
    print(f"  -> {out.relative_to(REPO)} ({out.stat().st_size / 2**20:.2f} MB, K={len(clusters)})")
    return {"benchmark": benchmark, "policy": policy, "task": task, "task_id": bundle["task_id"], "domain": bundle["domain"],
            "success_rate": bundle["success_rate"], "held_out": bundle["held_out"], "n_rollouts": len(rollouts), "K": len(clusters)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=DATASETS)
    ap.add_argument("--policy", required=True, choices=list(POLICY_DIR.values()))
    ap.add_argument("--task", nargs="*", help="task folder names; default: every task with embeddings")
    args = ap.parse_args()
    from src.prm.train_precog import PrecogProtoNet
    models = {name: PrecogProtoNet.load(path(args.benchmark)) for name, path in MODELS.items()}
    held_out = set(json.load(open(DATA / "caches" / "task_splits.json")).get(args.benchmark, []))
    emb_root = DATA / "embeddings" / args.benchmark / args.policy
    tasks = args.task or sorted(p.name for p in emb_root.iterdir() if (p / "action.npz").exists())
    index_path = OUT / "index.json"
    index = json.load(open(index_path)) if index_path.exists() else {"tasks": []}
    done = {(t["benchmark"], t["policy"], t["task"]): t for t in index["tasks"]}
    for task in tasks:
        done[(args.benchmark, args.policy, task)] = build(args.benchmark, args.policy, task, models, held_out)
    index["tasks"] = sorted(done.values(), key=lambda t: (t["benchmark"], t["policy"], t["task"]))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(index, open(index_path, "w"), indent=1)
    print(f"index: {len(index['tasks'])} tasks -> {index_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
