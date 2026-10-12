#!/usr/bin/env python3
"""Appendices C and F: the single-seed design study (tab:ablations,
tab:design-ablations), the mixed-pool replication, and the distance-rule test.

Protocol (fixed for every row): train on the Tau2-Bench/Qwen3-80B train-split
tasks (10 epochs x 400 episodes = 4,000 episodes, seed 0), evaluate on the
held-out test tasks under the standard support protocol (k=4 per class, 5
seeded draws) in three slices -- Within (tau2/qwen), X-model (tau2 under the
other two policies), X-domain (the six cells of the other two benchmarks).
Readout rows re-score the base checkpoint with alternative inference rules.

Results are cached per (regime, ablation, seed, variant) in
exp/results/protonet_ablations/results.json; checkpoints live in
$PRECOG_DATA/ckpts/ablations/{src,mix}_<ablation>_s0.pt.

    python scripts/design_study.py                 # print the paper tables from the cache
    python scripts/design_study.py --eval          # re-score every shipped checkpoint (no training)
    python scripts/design_study.py --train         # also meta-train any missing checkpoint
    python scripts/design_study.py --distance      # distance-rule test (App. F, last paragraph)

Note on naming: the paper's "Full configuration" (no action/observation type
flag, the shipped PrecogProtoNet default) is the study's `no_typebit` run; the
study's `full` run carries the flag and is the paper's "+ action/obs. flag" row.
"""
import argparse
import json
import time

import numpy as np

from _paths import ABL, CKPT_ABL, SPLITS, MMAP_DIR, DATASETS, MODELS, require

RESULTS_PATH = ABL / "results.json"
SRC = ("tau2", "qwen")
SETTINGS = {
    "S1_within": [SRC],
    "S2_xmodel": [("tau2", m) for m in MODELS if m != "qwen"],
    "S3_xdomain": [(ds, m) for ds in DATASETS if ds != "tau2" for m in MODELS],
    "S4_mixgen": [(ds, m) for ds in DATASETS for m in MODELS],
}
REGIME_SETTINGS = {"src": ["S1_within", "S2_xmodel", "S3_xdomain"], "mix": ["S4_mixgen"]}
PROTO_KWARGS = dict(hidden_dim=128, n_heads=4, n_layers=2, ffn_dim=512, dropout=0.3,
                    epochs=10, lr=1e-4, k_support=4, q_query=4, episodes_per_epoch=400,
                    lambda_adv=0.0, no_pos=True)
MIX_PROTO_OVERRIDES = dict(episodes_per_epoch=1000)
MIX_MAX_ROLLOUTS_PER_TASK = 30

# study name -> spec.  `flag` = action/observation type-flag column (the study's
# base class carried it; the shipped PrecogProtoNet default is no flag).
ABLATIONS = {
    "full":           dict(group="reference", flag=True),
    "no_typebit":     dict(group="input", flag=False),
    "act_only":       dict(group="input", flag=True, input_mode="act_only"),
    "res_only":       dict(group="input", flag=True, input_mode="res_only"),
    "with_pos":       dict(group="input", flag=True, proto=dict(no_pos=False)),
    "no_transformer": dict(group="arch", flag=True, proto=dict(n_layers=0)),
    "layers1":        dict(group="arch", flag=True, proto=dict(n_layers=1)),
    "layers4":        dict(group="arch", flag=True, proto=dict(n_layers=4)),
    "h32":            dict(group="arch", flag=True, proto=dict(hidden_dim=32)),
    "h256":           dict(group="arch", flag=True, proto=dict(hidden_dim=256)),
    "dropout0":       dict(group="arch", flag=True, proto=dict(dropout=0.0)),
    "attn_pool":      dict(group="arch", flag=True, pooling="attention"),
    "untrained":      dict(group="training", flag=True, proto=dict(epochs=0)),
    "epochs2":        dict(group="training", flag=True, proto=dict(epochs=2)),
    "epochs20":       dict(group="training", flag=True, proto=dict(epochs=20)),
    "k1q1":           dict(group="training", flag=True, proto=dict(k_support=1, q_query=1)),
    "k8q8":           dict(group="training", flag=True, proto=dict(k_support=8, q_query=8)),
    "prefix_train":   dict(group="training", flag=True, attrs=dict(prefix_training=True)),
    "adv_model":      dict(group="mix", flag=True, proto=dict(lambda_adv=0.1), adv="model", mix_only=True),
    "adv_dataset":    dict(group="mix", flag=True, proto=dict(lambda_adv=0.1), adv="dataset", mix_only=True),
}
EVAL_VARIANTS = {"metric=cosine": dict(metric="cosine"), "head=knn": dict(head="knn"),
                 "head=lr": dict(head="lr"), "stratified=False": dict(stratified=False)}

# paper rows: (label, results key)
TAB_ABLATIONS = [
    ("Full configuration", "src|no_typebit|s0|std"), ("+ action/obs. flag", "src|full|s0|std"),
    ("+ positional embeddings", "src|with_pos|s0|std"), ("    + permuted steps", "src|with_pos|s0|permute=True"),
    ("No transformer", "src|no_transformer|s0|std"), ("L=1 layer", "src|layers1|s0|std"),
    ("L=4 layers", "src|layers4|s0|std"), ("d'=32", "src|h32|s0|std"), ("d'=256", "src|h256|s0|std"),
    ("No dropout", "src|dropout0|s0|std"), ("Attention pooling", "src|attn_pool|s0|std"),
]
TAB_DESIGN = [
    ("Full configuration", "src|no_typebit|s0|std"), ("0.2x episodes", "src|epochs2|s0|std"),
    ("2x episodes", "src|epochs20|s0|std"), ("N_S=N_Q=1", "src|k1q1|s0|std"), ("N_S=N_Q=8", "src|k8q8|s0|std"),
    ("Prefix training", "src|prefix_train|s0|std"), ("Untrained encoder", "src|untrained|s0|std"),
    ("Cosine distance", "src|full|s0|metric=cosine"), ("kNN (k=5)", "src|full|s0|head=knn"),
    ("Logistic regression", "src|full|s0|head=lr"), ("Unstratified support", "src|full|s0|stratified=False"),
]
TAB_MIX = [("Full (study base, with flag)", "mix|full|s0|std"), ("+ positional embeddings", "mix|with_pos|s0|std"),
           ("No transformer", "mix|no_transformer|s0|std"), ("Untrained encoder", "mix|untrained|s0|std")]


def res_key(regime, name, seed=0, variant="std"):
    return f"{regime}|{name}|s{seed}|{variant}"


def _cell_mean(d):
    return float(np.mean(list(d.values()))) if d else np.nan


def setting_mean(res, setting):
    res = res.get("cells", res)
    vals = [_cell_mean(res.get(f"{ds}/{m}", {})) for ds, m in SETTINGS[setting]]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def print_tables(R):
    def row(label, key, settings):
        if key not in R:
            return f"  {label:26} " + " ".join(f"{'--':>8}" for _ in settings) + f"   (missing {key})"
        return f"  {label:26} " + " ".join(f"{setting_mean(R[key], s):8.3f}" for s in settings)
    s3 = REGIME_SETTINGS["src"]
    print("=== Table: encoder architecture ablations (App. C) -- Within / X-model / X-domain ===")
    for label, key in TAB_ABLATIONS:
        print(row(label, key, s3))
    print("\n=== Table: training-recipe and readout ablations (App. F) ===")
    for label, key in TAB_DESIGN:
        print(row(label, key, s3))
    print("\n=== Mixed pool of all nine cells (App. C text): held-out AUROC over all cells ===")
    for label, key in TAB_MIX:
        print(row(label, key, ["S4_mixgen"]))
    print("\n=== Per-task baselines under the same support protocol (k=4) ===")
    for mm in ("NC", "RBF", "FT1"):
        print(row(mm, res_key("none", mm, 0, "k4"), s3 + ["S4_mixgen"]))


def print_distance(R):
    print("=== Distance rule: unsquared vs squared Euclidean, matched pairs (App. F) ===")
    for pool, cells, label in (("tau2:q", [f"tau2/{m}" for m in MODELS], "single-policy pool tau2:q, in-domain (3 tau2 cells)"),
                               ("bfclv3:g+q+d", [f"bfclv3/{m}" for m in MODELS], "all-policy pool bfclv3:g+q+d, in-domain (3 bfclv3 cells)")):
        vals = {}
        for dist in ("euclid", "sqeuclid"):
            k = f"disttest|{pool}|{dist}|s0|std"
            if k not in R:
                print(f"  missing {k}; run with --distance to compute"); break
            c = R[k]["cells"]
            vals[dist] = np.mean([_cell_mean(c[x]) for x in cells])
        if len(vals) == 2:
            print(f"  {label}: unsquared {vals['euclid']:.3f} vs squared {vals['sqeuclid']:.3f}")


# ----------------------------------------------------------------- evaluation / training
def _load_W():
    import pool_traj_worker as W
    W.load_all_cells()
    return W, {ds: set(v) for ds, v in json.load(open(SPLITS)).items()}


def _heads():
    from sklearn.linear_model import LogisticRegression
    def proto(Es, ys, Eq, metric="euclid"):
        if metric == "cosine":
            n = lambda X: X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)
            Es, Eq = n(Es), n(Eq)
        cp, cf = Es[ys == 1].mean(0), Es[ys == 0].mean(0)
        return np.linalg.norm(Eq - cf, axis=1) - np.linalg.norm(Eq - cp, axis=1)
    def knn(Es, ys, Eq, k=5):
        D = ((Eq[:, None, :] - Es[None, :, :]) ** 2).sum(-1)
        kk = min(k, len(ys))
        return np.array([ys[np.argpartition(row, kk - 1)[:kk]].mean() for row in D])
    def lr(Es, ys, Eq):
        if len(set(ys)) < 2:
            return np.zeros(len(Eq))
        return LogisticRegression(max_iter=1000, C=1.0).fit(Es, ys).predict_proba(Eq)[:, 1]
    return proto, knn, lr


def support_draws(W, t, k, n_repeats, stratified=True):
    rng = np.random.RandomState(W.SEED)
    y = np.array([t["y"][r] for r in t["rids"]])
    pi0, fi0 = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n_repeats):
        if stratified:
            pi, fi = rng.permutation(pi0), rng.permutation(fi0)
            sup, qry = np.concatenate([pi[:k], fi[:k]]), np.concatenate([pi[k:], fi[k:]])
        else:
            perm = rng.permutation(len(y)); sup, qry = perm[:2 * k], perm[2 * k:]
        if len(set(y[sup])) < 2 or len(set(y[qry])) < 2:
            continue
        out.append((sup, qry))
    return out


def eval_task(W, prm, t, head="proto", metric="euclid", permute=False, stratified=True):
    from sklearn.metrics import roc_auc_score
    proto, knn, lr = _heads()
    rng = np.random.RandomState(W.SEED)
    E = []
    for r in t["rids"]:
        a, s = t["aseq"][int(r)], t["rseq"][int(r)]
        if permute and len(a) > 1:
            p = rng.permutation(len(a)); a, s = a[p], s[p]
        E.append(prm._encode_rollout(a, s))
    E = np.stack(E)
    y = np.array([t["y"][r] for r in t["rids"]])
    hd = {"proto": (lambda s_, ys, q: proto(s_, ys, q, metric)), "knn": knn, "lr": lr}[head]
    aucs = [roc_auc_score(y[q], hd(E[s_], y[s_], E[q]))
            for s_, q in support_draws(W, t, W.KSUP_EVAL, W.CMP_REPEATS, stratified)]
    return float(np.mean(aucs)) if aucs else np.nan


def eval_on(W, test, prm, cells, **variant):
    out = {}
    for ds, m in cells:
        per = {}
        for t in W.eval_tasks_for(ds, m, test):
            a = eval_task(W, prm, t, **variant)
            if not np.isnan(a):
                per[t["label"]] = a
        out[f"{ds}/{m}"] = per
        if per:
            print(f"    {ds}/{m}: {len(per)} tasks AUROC={np.mean(list(per.values())):.3f}", flush=True)
    return out


def train_or_load(W, test, name, regime, seed, allow_train):
    import torch
    from src.prm.train_precog import PrecogProtoNet
    spec = ABLATIONS[name]
    ckpt = CKPT_ABL / f"{regime}_{name}_s{seed}.pt"
    if ckpt.exists():
        prm = PrecogProtoNet.load(str(ckpt))
        prm.input_mode = spec.get("input_mode", "both")
        return prm
    if not allow_train:
        return None
    cells = [SRC] if regime == "src" else SETTINGS["S4_mixgen"]
    act, res, lab, tid, mids, dsids = W._gather(cells, test, max_per_task=None if regime == "src"
                                                else MIX_MAX_ROLLOUTS_PER_TASK)
    kw = {**PROTO_KWARGS, **(MIX_PROTO_OVERRIDES if regime == "mix" else {}), **spec.get("proto", {})}
    torch.manual_seed(seed); np.random.seed(seed)
    prm = PrecogProtoNet(**kw, use_type_flag=spec.get("flag", False))
    prm.input_mode = spec.get("input_mode", "both")
    if "pooling" in spec:
        prm.pooling = spec["pooling"]
    for a, v in spec.get("attrs", {}).items():
        setattr(prm, a, v)
    ids = {"model": mids, "dataset": dsids}.get(spec.get("adv"))
    t0 = time.time()
    prm.fit_from_embeddings(act, res, lab, tid, model_ids=ids)
    CKPT_ABL.mkdir(parents=True, exist_ok=True)
    prm.save(str(ckpt))
    print(f"  trained {regime}/{name}/s{seed} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return prm


def run_study(R, allow_train, force, only=None):
    require(MMAP_DIR, "mmap caches", "caches")
    W, test = _load_W()
    jobs = []
    for name, spec in ABLATIONS.items():
        if not spec.get("mix_only"):
            jobs.append(("src", name, "std", {}))
    for v, kw in EVAL_VARIANTS.items():
        jobs.append(("src", "full", v, kw))
    jobs.append(("src", "with_pos", "permute=True", dict(permute=True)))
    for name in ("full", "untrained", "act_only", "res_only", "with_pos", "no_transformer",
                 "adv_model", "adv_dataset"):
        jobs.append(("mix", name, "std", {}))
    for regime, name, vtag, kw in jobs:
        if only and name not in only:
            continue
        key = res_key(regime, name, 0, vtag)
        if key in R and not force:
            continue
        prm = train_or_load(W, test, name, regime, 0, allow_train)
        if prm is None:
            print(f"  [skip] no checkpoint {regime}_{name}_s0.pt (use --train)"); continue
        cells = list(dict.fromkeys(sum((SETTINGS[s] for s in REGIME_SETTINGS[regime]), [])))
        print(f"== eval {key} ==", flush=True)
        R[key] = eval_on(W, test, prm, cells, **kw)
        json.dump(R, open(RESULTS_PATH, "w"))


def run_distance(R, force):
    import torch
    from src.prm.train_precog import PrecogProtoNet
    from sklearn.metrics import roc_auc_score
    require(MMAP_DIR, "mmap caches", "caches")
    W, test = _load_W()
    for pool in ("tau2:q", "bfclv3:g+q+d"):
        gathered = None
        for dist in ("euclid", "sqeuclid"):
            key = f"disttest|{pool}|{dist}|s0|std"
            if key in R and not force:
                continue
            if gathered is None:
                gathered = W._gather(W.POOLS[pool], test, max_per_task=None)
            act, res, lab, tid, *_ = gathered
            torch.manual_seed(0); np.random.seed(0)
            prm = PrecogProtoNet(**PROTO_KWARGS, distance=dist)
            losses = []
            prm.fit_from_embeddings(act, res, lab, tid, epoch_callback=lambda p, e, l: losses.append(round(l, 4)))
            out = {"loss": losses, "cells": {}}
            for ds, m in SETTINGS["S4_mixgen"]:
                per = {}
                for t in W.eval_tasks_for(ds, m, test):
                    E = np.stack([prm._encode_rollout(t["aseq"][int(r)], t["rseq"][int(r)]) for r in t["rids"]])
                    y = np.array([t["y"][r] for r in t["rids"]])
                    aucs = []
                    for s_, q in support_draws(W, t, W.KSUP_EVAL, W.CMP_REPEATS):
                        cp, cf = E[s_][y[s_] == 1].mean(0), E[s_][y[s_] == 0].mean(0)
                        dp, df_ = ((E[q] - cp) ** 2).sum(1), ((E[q] - cf) ** 2).sum(1)
                        if dist == "euclid":
                            dp, df_ = np.sqrt(dp), np.sqrt(df_)
                        aucs.append(roc_auc_score(y[q], df_ - dp))
                    if aucs:
                        per[t["label"]] = float(np.mean(aucs))
                out["cells"][f"{ds}/{m}"] = per
            R[key] = out
            json.dump(R, open(RESULTS_PATH, "w"))
            print(f"  done {key}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="store_true", help="re-score shipped checkpoints (overwrites cache)")
    ap.add_argument("--train", action="store_true", help="meta-train missing checkpoints, then evaluate")
    ap.add_argument("--distance", action="store_true", help="run the distance-rule test if not cached")
    ap.add_argument("--only", nargs="*", help="restrict --eval/--train to these ablation names")
    args = ap.parse_args()
    R = json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}
    if args.eval or args.train:
        run_study(R, allow_train=args.train, force=args.eval, only=args.only)
    if args.distance:
        run_distance(R, force=False)
    print_tables(R)
    print()
    print_distance(R)


if __name__ == "__main__":
    main()
