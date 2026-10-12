#!/usr/bin/env python3
"""Precog-config pool sweep: 25 pools x seeds, fixed budget, epoch-resolved TEST curves.

Final results run for the outcome-space analysis section: PrecogProtoNet defaults
(no positions, no type flag, unsquared Euclidean, lazy tokens), all rollouts (no cap),
NO validation split — every train task trains.

Protocol:
  * episodes_per_epoch=100, epochs=100 (10k-episode ceiling = slowest observed
    saturation; same LR schedule/episode sequence as a 400x25 run, and the same
    100-episode unit as the dense-resolution trajectories).
  * Every EVAL_EVERY epochs, the full 9-cell TEST grid (matched-support AUROC,
    per task) is evaluated with batched encoding and stored -> per-job test curve.
    Peaks/saturation are read off the curves; there is no early stopping and no
    checkpoint selection. REPORTING DISCIPLINE: quote curves or fixed-budget points,
    never per-pool test-curve maxima (that is test-set selection).
  * Final-epoch weights are checkpointed (any intermediate epoch is reproducible by
    rerunning the job: same seed => same trajectory).
  * One part file per (pool, seed) under precog_parts/ -> no write races.

Usage:
  python scripts/precog_sweep.py --plan 2         # split remaining jobs into 2 commands
  python scripts/precog_sweep.py 'tau2:q@s0' ...  # run job tokens sequentially
  python scripts/precog_sweep.py --list
  # extra CPU worker: ... --device cpu --threads 4
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
from sklearn.metrics import roc_auc_score
import torch

from scripts._paths import ABL, CKPT_PRECOG  # noqa: E402
SEEDS = [0, 1, 2]
K = 4

def _fname(pool, seed):
    return pool.replace(":", "_").replace("+", "-") + f"_s{seed}"

def proto_auc(prm, t, batch=64):
    E = W._encode_all(prm, t, batch=batch)
    y = np.array([t["y"][r] for r in t["rids"]])
    aucs = []
    for s, q in W.support_draws(t, K):
        cp, cf = E[s][y[s] == 1].mean(0), E[s][y[s] == 0].mean(0)
        sc = np.linalg.norm(E[q] - cf, axis=1) - np.linalg.norm(E[q] - cp, axis=1)
        aucs.append(roc_auc_score(y[q], sc))
    return float(np.mean(aucs)) if aucs else np.nan

def run_job(pool, seed, args, test_labels):
    part = args.parts_dir / f"{_fname(pool, seed)}.json"
    if part.exists() and not args.force:
        print(f"[skip] {pool}@s{seed} (part exists)", flush=True); return
    act, res, lab, tid = [], [], [], []
    for ds, m in W.POOLS[pool]:
        for t in W.train_tasks_for(ds, m, test_labels):
            for r in t["rids"]:
                r = int(r)
                act.append(t["aseq"][r]); res.append(t["rseq"][r])
                lab.append(t["y"][r]); tid.append(f"{ds}|{m}|{t['label']}")
    print(f"== {pool}@s{seed}: {len(set(tid))} train groups, {len(lab)} rollouts ==",
          flush=True)

    def eval_grid(p):
        out = {}
        for ds, m in W.cells_grid():
            per = {}
            for t in W.eval_tasks_for(ds, m, test_labels):
                a = proto_auc(p, t, batch=args.eval_batch)
                if not np.isnan(a): per[t["label"]] = a
            out[f"{ds}/{m}"] = per
        return out

    torch.manual_seed(seed); np.random.seed(seed)
    prm = PrecogProtoNet(**{**W.PROTO_KWARGS,
                            "epochs": args.max_epochs,
                            "episodes_per_epoch": args.episodes})
    curve = {}

    def _cb(p, epoch, loss):
        if epoch % args.eval_every != 0 and epoch != args.max_epochs:
            return False
        curve[str(epoch)] = eval_grid(p)
        gm = np.mean([np.mean(list(d.values())) for d in curve[str(epoch)].values() if d])
        print(f"    epoch {epoch:3d} ({epoch * args.episodes} eps)  loss={loss:.4f}  "
              f"test_grid_mean={gm:.4f}", flush=True)
        return False   # no early stopping — fixed budget, curves are the product

    t0 = time.time()
    prm.fit_from_embeddings(act, res, lab, tid, epoch_callback=_cb)
    prm.save(str(args.ckpt_dir / f"precog_{_fname(pool, seed)}.pt"))

    args.parts_dir.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".tmp")
    json.dump({"pool": pool, "seed": seed, "config": "precog_v3_noval",
               "epochs": args.max_epochs, "episodes_per_epoch": args.episodes,
               "eval_every": args.eval_every, "test_curve": curve}, open(tmp, "w"))
    os.replace(tmp, part)
    print(f"[done] {pool}@s{seed} in {(time.time()-t0)/60:.1f} min", flush=True)

def jobs_remaining(args):
    out = []
    for pool in sorted(W.POOLS, key=lambda p: -len(W.POOLS[p])):  # big pools first
        for seed in SEEDS:
            if not (args.parts_dir / f"{_fname(pool, seed)}.json").exists():
                out.append(f"{pool}@s{seed}")
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", help="job tokens like 'tau2:q@s0'")
    ap.add_argument("--plan", type=int, help="split remaining jobs into N worker commands")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-batch", type=int, default=64,
                    help="batched rollout encoding for test evals (0 = per-rollout; "
                         "verified numerically identical)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto",
                    help="cpu lets extra workers run without MPS contention")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU thread cap per worker (recommend 4 for cpu workers)")
    ap.add_argument("--parts-dir", type=Path, default=ABL / "precog_parts")
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_PRECOG)
    args = ap.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.device != "auto":
        from src.prm.cluster_sequence import ProtoNetPRM
        dev = args.device
        ProtoNetPRM._select_device = staticmethod(lambda: torch.device(dev))
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.list:
        rem = jobs_remaining(args)
        print(f"{75 - len(rem)}/75 done; remaining: {rem}"); return
    if args.plan:
        rem = jobs_remaining(args)
        buckets = [rem[i::args.plan] for i in range(args.plan)]
        print(f"# {len(rem)} jobs -> {args.plan} workers:")
        for b in buckets:
            if b: print("python scripts/precog_sweep.py " + " ".join(f"'{j}'" for j in b))
        return
    if not args.jobs:
        ap.print_help(); return

    test_labels = {ds: set(v) for ds, v in json.load(open(W.SPLITS_PATH)).items()}
    W.load_all_cells()
    for tok in args.jobs:
        pool, seed = tok.rsplit("@s", 1)
        if pool not in W.POOLS:
            sys.exit(f"unknown pool: {pool}")
        run_job(pool, int(seed), args, test_labels)

if __name__ == "__main__":
    main()
