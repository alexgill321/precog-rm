#!/usr/bin/env python3
"""Support-size scaling: precog benchmark specialists vs per-task baselines.

For each benchmark, evaluates the {bench}:g+q+d precog checkpoints (all seeds) and the
NC / RBF / FT1 baselines at k in {1,2,4,8} support rollouts per class, on a FIXED
cohort of held-out tasks (>= max(k)+1 rollouts per class, so the same tasks are scored
at every k). Identical seeded draws per (task, k) across all methods. Eval-only.

  python scripts/support_sweep.py                 # all benchmarks, MPS
  python scripts/support_sweep.py --device cpu    # avoid MPS contention with training
Results -> exp/results/protonet_ablations/support_sweep.json (incremental, resumable)
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)
import scripts.pool_traj_worker as W
from src.prm.train_precog import PrecogProtoNet
from src.prm.cluster_sequence import FastTextPRM, _aggregate_scores, ProtoNetPRM
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import torch

ABL = REPO / "exp/results/protonet_ablations"
OUT = ABL / "support_sweep.json"
KS = [1, 2, 4, 8]
REPEATS, SEED = 5, 42
SEEDS = [0, 1, 2]

def _san(P):
    return np.nan_to_num(np.stack(P), nan=0., posinf=0., neginf=0.).astype(np.float64)

def draws(t, k):
    rng = np.random.RandomState(SEED)   # same stream per task; k slices the same perms
    y = np.array([t["y"][r] for r in t["rids"]])
    pi0, fi0 = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(REPEATS):
        pi, fi = rng.permutation(pi0), rng.permutation(fi0)
        sup, qry = np.concatenate([pi[:k], fi[:k]]), np.concatenate([pi[k:], fi[k:]])
        if len(set(y[sup])) < 2 or len(set(y[qry])) < 2: continue
        out.append((sup, qry))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--eval-batch", type=int, default=64)
    args = ap.parse_args()
    if args.device != "auto":
        dev = args.device
        ProtoNetPRM._select_device = staticmethod(lambda: torch.device(dev))

    test_labels = {ds: set(v) for ds, v in json.load(open(W.SPLITS_PATH)).items()}
    W.load_all_cells()
    res = json.load(open(OUT)) if OUT.exists() else {}

    for ds in W.DATASETS:
        prms = {}
        for m in W.MODELS:
            cohort = [t for t in W.eval_tasks_for(ds, m, test_labels)
                      if min(int(sum(t["y"].values())),
                             len(t["y"]) - int(sum(t["y"].values()))) >= max(KS) + 1]
            ck = f"{ds}/{m}"
            if ck in res:
                print(f"[cached] {ck}"); continue
            print(f"== {ck}: cohort {len(cohort)} tasks ==", flush=True)
            if not cohort:
                res[ck] = {}; continue
            t0 = time.time()
            per = {}   # {k: {method: {task: auc}}}
            for k in KS:
                per[str(k)] = {mm: {} for mm in
                               ["ProtoNet"] + [f"ProtoNet_s{s}" for s in SEEDS]
                               + ["NC", "RBF", "FT1"]}
            E_seeds = {}
            for s in SEEDS:
                if s not in prms:
                    prms[s] = PrecogProtoNet.load(
                        str(ABL / "ckpts" / f"precog_{ds.replace(':','_')}_g-q-d_s{s}.pt"))
            for t in cohort:
                y = np.array([t["y"][r] for r in t["rids"]])
                P = _san([t["pooled"][r] for r in t["rids"]])
                E = {s: W._encode_all(prms[s], t, batch=args.eval_batch) for s in SEEDS}
                for k in KS:
                    dd = draws(t, k)
                    if not dd: continue
                    acc = {mm: [] for mm in per[str(k)]}
                    for sup, qry in dd:
                        yq = y[qry]
                        # baselines
                        cp, cf = P[sup][y[sup]==1].mean(0), P[sup][y[sup]==0].mean(0)
                        cp /= max(np.linalg.norm(cp),1e-9); cf /= max(np.linalg.norm(cf),1e-9)
                        acc["NC"].append(roc_auc_score(yq, P[qry]@cp - P[qry]@cf))
                        if len(set(y[sup])) == 2:
                            svm = SVC(kernel="rbf", gamma="scale", C=1.0).fit(P[sup], y[sup])
                            acc["RBF"].append(roc_auc_score(yq, svm.decision_function(P[qry])))
                        sup_r = [t["rids"][i] for i in sup]; qry_r = [t["rids"][i] for i in qry]
                        ft = FastTextPRM(wordNgrams=1, dim=20)
                        ft.fit({int(r): t["path"][int(r)] for r in sup_r},
                               {int(r): t["y"][int(r)] for r in sup_r}, t["K"])
                        acc["FT1"].append(roc_auc_score(
                            yq, [_aggregate_scores(ft.score_path(t["path"][int(r)], rid=int(r)),
                                                   "final") for r in qry_r]))
                        # protonet per seed
                        for s in SEEDS:
                            Es = E[s]
                            pp, ff = Es[sup][y[sup]==1].mean(0), Es[sup][y[sup]==0].mean(0)
                            sc = (np.linalg.norm(Es[qry]-ff,axis=1)
                                  - np.linalg.norm(Es[qry]-pp,axis=1))
                            acc[f"ProtoNet_s{s}"].append(roc_auc_score(yq, sc))
                    for mm, v in acc.items():
                        if v and mm != "ProtoNet":
                            per[str(k)][mm][t["label"]] = float(np.mean(v))
                    pn = [np.mean(acc[f"ProtoNet_s{s}"]) for s in SEEDS
                          if acc[f"ProtoNet_s{s}"]]
                    if pn: per[str(k)]["ProtoNet"][t["label"]] = float(np.mean(pn))
            res[ck] = per
            json.dump(res, open(OUT, "w"))
            print(f"[done] {ck} in {(time.time()-t0)/60:.1f} min", flush=True)

    # summary
    print(f"\n{'cell':<24}{'k':>3}" + "".join(f"{mm:>10}" for mm in ["ProtoNet","NC","RBF","FT1"]))
    for ck, per in res.items():
        for k in map(str, KS):
            if k not in per or not per[k].get("ProtoNet"): continue
            row = "".join(f"{np.mean(list(per[k][mm].values())):>10.3f}"
                          if per[k].get(mm) else f"{'--':>10}"
                          for mm in ["ProtoNet","NC","RBF","FT1"])
            print(f"{ck:<24}{k:>3}" + row)

if __name__ == "__main__":
    main()
