#!/usr/bin/env python3
"""Merged main-table: per-task 3-fold CV on TEST-SPLIT tasks, baselines + generalist.

Protocol (matches separability_questions.ipynb Table 1 exactly where applicable):
  * folds: KFold(3, shuffle=True, random_state=42) over each task's rids
  * baselines fit on the train fold (~33 rollouts): NC / RBF on pooled,
    fastText n in {1,2,5} on clustered state-ID paths
  * PrecogRM generalist (all-benchmark all-model pool, 3 seeds) uses the SAME
    train-fold rollouts as its support: prototypes from train fold, scores test fold
  * per-task AUROC = mean over valid folds; cell mean over tasks; fold-level std
    computed as in the notebook (mean across tasks within fold -> std across folds)

Results -> exp/results/protonet_ablations/merged_cv_table.json (incremental per cell)
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)
import scripts.pool_traj_worker as W
from src.prm.train_precog import PrecogProtoNet
from src.prm.cluster_sequence import FastTextPRM, _aggregate_scores
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

ABL = REPO / "exp/results/protonet_ablations"
OUT = ABL / "merged_cv_table.json"
N_FOLDS, SEED = 3, 42
PN_SEEDS = [0, 1, 2]
GENERALIST = "agentclinic-bfclv3-tau2_all"
METHODS = ["NC", "RBF", "FT1", "FT2", "FT5"] + [f"PN_s{s}" for s in PN_SEEDS]

def _san(P):
    return np.nan_to_num(np.stack(P), nan=0., posinf=0., neginf=0.).astype(np.float64)

def main():
    test_labels = {ds: set(v) for ds, v in json.load(open(W.SPLITS_PATH)).items()}
    W.load_all_cells()
    prms = {s: PrecogProtoNet.load(str(ABL / "ckpts" / f"precog_{GENERALIST}_s{s}.pt"))
            for s in PN_SEEDS}
    res = json.load(open(OUT)) if OUT.exists() else {}

    for ds in W.DATASETS:
        for m in W.MODELS:
            ck = f"{ds}/{m}"
            if ck in res:
                print(f"[cached] {ck}", flush=True); continue
            tasks = W.eval_tasks_for(ds, m, test_labels)
            print(f"== {ck}: {len(tasks)} test tasks ==", flush=True)
            t0 = time.time()
            # per[method][task_label] = list of per-fold aucs (nan = skipped fold)
            per = {mm: {} for mm in METHODS}
            for t in tasks:
                rids = t["rids"]
                y = np.array([t["y"][r] for r in rids])
                P = _san([t["pooled"][r] for r in rids])
                E = {s: W._encode_all(prms[s], t, batch=64) for s in PN_SEEDS}
                folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(rids))
                fa = {mm: [] for mm in METHODS}
                for tr, te in folds:
                    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                        for mm in METHODS: fa[mm].append(np.nan)
                        continue
                    yt = y[te]
                    cp, cf = P[tr][y[tr] == 1].mean(0), P[tr][y[tr] == 0].mean(0)
                    cp /= max(np.linalg.norm(cp), 1e-9); cf /= max(np.linalg.norm(cf), 1e-9)
                    fa["NC"].append(roc_auc_score(yt, P[te] @ cp - P[te] @ cf))
                    svm = SVC(kernel="rbf", gamma="scale", C=1.0).fit(P[tr], y[tr])
                    fa["RBF"].append(roc_auc_score(yt, svm.decision_function(P[te])))
                    tr_r = [int(rids[i]) for i in tr]; te_r = [int(rids[i]) for i in te]
                    for n in (1, 2, 5):
                        ft = FastTextPRM(wordNgrams=n, dim=20)
                        ft.fit({r: t["path"][r] for r in tr_r},
                               {r: t["y"][r] for r in tr_r}, t["K"])
                        fa[f"FT{n}"].append(roc_auc_score(
                            yt, [_aggregate_scores(ft.score_path(t["path"][r], rid=r),
                                                   "final") for r in te_r]))
                    for s in PN_SEEDS:
                        Es = E[s]
                        pp, ff = Es[tr][y[tr] == 1].mean(0), Es[tr][y[tr] == 0].mean(0)
                        sc = (np.linalg.norm(Es[te] - ff, axis=1)
                              - np.linalg.norm(Es[te] - pp, axis=1))
                        fa[f"PN_s{s}"].append(roc_auc_score(yt, sc))
                for mm in METHODS:
                    per[mm][t["label"]] = fa[mm]
            res[ck] = per
            json.dump(res, open(OUT, "w"))
            print(f"[done] {ck} in {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- summary: cell mean over tasks (of per-task fold-mean) + fold-level std ----
    print(f"\n{'cell':<26}" + "".join(f"{mm:>12}" for mm in ["NC","RBF","FT1","FT2","FT5","PN"]))
    for ck, per in res.items():
        row = ""
        for mm in ["NC", "RBF", "FT1", "FT2", "FT5"]:
            M = np.array(list(per[mm].values()), dtype=float)   # (tasks, folds)
            mean = np.nanmean(np.nanmean(M, axis=1))
            fstd = np.nanstd(np.nanmean(M, axis=0))
            row += f"{mean:.3f}±{fstd:.3f} ".rjust(12)
        pn = np.stack([np.array(list(per[f"PN_s{s}"].values()), dtype=float)
                       for s in PN_SEEDS])                       # (seeds, tasks, folds)
        mean = np.nanmean(np.nanmean(np.nanmean(pn, axis=2), axis=1))
        fstd = np.nanstd(np.nanmean(pn, axis=1))                 # std over seeds x folds
        row += f"{mean:.3f}±{fstd:.3f} ".rjust(12)
        print(f"{ck:<26}" + row)

if __name__ == "__main__":
    main()
