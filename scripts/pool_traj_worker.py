#!/usr/bin/env python3
"""Parallel worker for the protonet_ablations pool-trajectory sweep.

Standalone version of the notebook's `run_trajectory` (notebooks/protonet_ablations.ipynb),
built for running several pools concurrently on one machine:

  * embeddings are read from mmap caches (one physical copy shared by all workers via the
    OS page cache) instead of per-process pickle loads;
  * each run writes its trajectory to its own part file under traj_parts/ (no shared-JSON
    race); fold parts into trajectories.json with --merge when no writers are active.

Protocol constants (splits, support draws, eval slice, PROTO_KWARGS) mirror the notebook —
keep them in sync. Training hyperparameters are flags so they can match the live kernel.

Typical use:
  python scripts/pool_traj_worker.py --convert          # once: slim pickles -> mmap caches
  python scripts/pool_traj_worker.py --list             # pool status
  python scripts/pool_traj_worker.py --plan 2           # split remaining pools into 2 commands
  python scripts/pool_traj_worker.py tau2:q tau2:g+d    # run these pools sequentially
  python scripts/pool_traj_worker.py --merge            # fold part files into trajectories.json
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, pickle, sys, time, zlib
from itertools import combinations
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# ==================== constants mirroring the notebook (keep in sync) ====================
DATASETS = ["agentclinic", "bfclv3", "tau2"]
MODELS = ["gpt5mini", "qwen", "deepseek"]
SEED = 42
KSUP_EVAL, CMP_REPEATS = 4, 5
MIN_TRAIN_PC = 8
AC_EVAL_SUBSET = "tool"
AC_DIR = REPO / "benchmark_refs/AgentClinic"
_AC_SUBSET_FILES = {
    "tool":    {"gpt5mini": "toolreliant_26.txt",
                "deepseek": "toolreliant_deepseek_fdr_mixed.txt",
                "qwen":     "toolreliant_qwen_fdr_mixed.txt"},
    "nontool": {"gpt5mini": "nontool_mixed.txt",
                "deepseek": "nontool_deepseek_mixed.txt",
                "qwen":     "nontool_qwen_mixed.txt"},
}
PROTO_KWARGS = dict(hidden_dim=128, n_heads=4, n_layers=2, ffn_dim=512, dropout=0.3,
                    epochs=10, lr=1e-4, k_support=4, q_query=4, episodes_per_epoch=400,
                    lambda_adv=0.0, no_pos=True)
ABL_DIR = REPO / "exp/results/protonet_ablations"
SLIM_DIR = ABL_DIR / "slim"
MMAP_DIR = ABL_DIR / "mmap"
SPLITS_PATH = ABL_DIR / "task_splits.json"

# minimal ablation registry (trajectories default to "full"; others available via --ablation)
# NOTE: PrecogProtoNet defaults are no_pos=True and use_type_flag=False, so "full" here
# trains WITHOUT positional embeddings and WITHOUT the action/response flag; the
# "with_*" entries turn each back on.
ABLATIONS = {
    "full":           dict(),
    "act_only":       dict(input_mode="act_only"),
    "res_only":       dict(input_mode="res_only"),
    "with_typebit":   dict(attrs=dict(use_type_flag=True)),
    "with_pos":       dict(proto=dict(no_pos=False)),
    "no_transformer": dict(proto=dict(n_layers=0)),
    "layers1":        dict(proto=dict(n_layers=1)),
    "layers4":        dict(proto=dict(n_layers=4)),
    "h32":            dict(proto=dict(hidden_dim=32)),
    "h256":           dict(proto=dict(hidden_dim=256)),
    "dropout0":       dict(proto=dict(dropout=0.0)),
    "attn_pool":      dict(pooling="attention"),
    "k1q1":           dict(proto=dict(k_support=1, q_query=1)),
    "k8q8":           dict(proto=dict(k_support=8, q_query=8)),
    "prefix_train":   dict(attrs=dict(prefix_training=True)),
    "adv_model":      dict(proto=dict(lambda_adv=0.1), adv="model"),
    "adv_dataset":    dict(proto=dict(lambda_adv=0.1), adv="dataset"),
}

def cells_grid():
    return [(ds, m) for ds in DATASETS for m in MODELS]

POOLS = {}
_ABBR = {"gpt5mini": "g", "qwen": "q", "deepseek": "d"}
for _ds in DATASETS:
    for _r in (1, 2, 3):
        for _combo in combinations(MODELS, _r):
            POOLS[f"{_ds}:{'+'.join(_ABBR[m] for m in _combo)}"] = [(_ds, m) for m in _combo]
for _r in (2, 3):
    for _combo in combinations(DATASETS, _r):
        POOLS[f"{'+'.join(_combo)}:all"] = [(ds, m) for ds in _combo for m in MODELS]

# ==================== mmap cache: convert + load ====================

def _mmap_paths(ds, m):
    return {s: MMAP_DIR / f"{ds}_{m}_{s}.npy" for s in ("act", "res", "pooled")} | \
           {"meta": MMAP_DIR / f"{ds}_{m}_meta.pkl"}

def convert_cell(ds, m, force=False):
    paths = _mmap_paths(ds, m)
    if not force and all(p.exists() for p in paths.values()):
        return
    MMAP_DIR.mkdir(parents=True, exist_ok=True)
    cell = pickle.load(open(SLIM_DIR / f"{ds}_{m}.pkl", "rb"))
    acts, ress, pooleds, meta = [], [], [], []
    off = pidx0 = 0
    for t in cell:
        offs, pidx = {}, {}
        for i, r in enumerate(t["rids"]):
            a = np.ascontiguousarray(t["aseq"][r], dtype=np.float16)
            offs[int(r)] = (off, len(a)); off += len(a)
            pidx[int(r)] = pidx0 + i
            acts.append(a)
            ress.append(np.ascontiguousarray(t["rseq"][r], dtype=np.float16))
            pooleds.append(np.asarray(t["pooled"][r], dtype=np.float32))
        pidx0 += len(t["rids"])
        meta.append({"label": t["label"], "K": t["K"], "sr": t["sr"],
                     "rids": [int(r) for r in t["rids"]],
                     "y": {int(r): int(v) for r, v in t["y"].items()},
                     "path": {int(r): list(p) for r, p in t["path"].items()},
                     "offs": offs, "pidx": pidx})
    np.save(paths["act"], np.concatenate(acts)); np.save(paths["res"], np.concatenate(ress))
    np.save(paths["pooled"], np.stack(pooleds))
    pickle.dump(meta, open(paths["meta"], "wb"))
    # verify one task round-trips exactly
    chk = load_cell(ds, m)
    t0, c0 = cell[0], chk[0]
    r0 = t0["rids"][0]
    assert np.array_equal(np.asarray(t0["aseq"][r0]), np.asarray(c0["aseq"][int(r0)]))
    assert np.array_equal(np.asarray(t0["rseq"][r0]), np.asarray(c0["rseq"][int(r0)]))
    assert np.allclose(np.asarray(t0["pooled"][r0]), np.asarray(c0["pooled"][int(r0)]))
    print(f"  converted {ds}/{m}: {len(cell)} tasks, {off} steps "
          f"({paths['act'].stat().st_size / 2**20:.0f} MB/channel)")

def load_cell(ds, m):
    """Task dicts in the notebook's shape, with aseq/rseq/pooled as zero-copy views into
    shared mmap arrays. Multiple processes share one physical copy via the page cache."""
    paths = _mmap_paths(ds, m)
    A = np.load(paths["act"], mmap_mode="r")
    R = np.load(paths["res"], mmap_mode="r")
    P = np.load(paths["pooled"], mmap_mode="r")
    cell = []
    for t in pickle.load(open(paths["meta"], "rb")):
        cell.append({"label": t["label"], "K": t["K"], "sr": t["sr"],
                     "rids": np.array(t["rids"]), "y": t["y"], "path": t["path"],
                     "aseq": {r: A[s:s + L] for r, (s, L) in t["offs"].items()},
                     "rseq": {r: R[s:s + L] for r, (s, L) in t["offs"].items()},
                     "pooled": {r: P[i] for r, i in t["pidx"].items()}})
    return cell

# ==================== eligibility / splits / eval tasks (mirrors notebook) ====================
CELLS = {}

def load_all_cells():
    for ds, m in cells_grid():
        CELLS[(ds, m)] = load_cell(ds, m)

def _class_counts(t): p = int(sum(t["y"].values())); return p, len(t["y"]) - p
def _train_ok(t): p, f = _class_counts(t); return p >= MIN_TRAIN_PC and f >= MIN_TRAIN_PC
def _test_ok(t, k=KSUP_EVAL): p, f = _class_counts(t); return p >= k + 1 and f >= k + 1

def ac_subset_ids(kind, model):
    fn = _AC_SUBSET_FILES[kind][model]
    return {int(x) for x in (AC_DIR / fn).read_text().split(",") if x.strip()}

def cell_key(ds, m): return f"{ds}/{m}"

def train_tasks_for(ds, m, test_labels):
    return [t for t in CELLS[(ds, m)] if t["label"] not in test_labels[ds] and _train_ok(t)]

def eval_tasks_for(ds, m, test_labels, k=KSUP_EVAL):
    pool = [t for t in CELLS[(ds, m)] if t["label"] in test_labels[ds] and _test_ok(t, k)]
    if ds == "agentclinic" and AC_EVAL_SUBSET != "all":
        ids = ac_subset_ids(AC_EVAL_SUBSET, m)
        pool = [t for t in pool if int(t["label"].split("/")[-1]) in ids]
    return pool

# ==================== model: lazy-token trainer (no per-pool token buffer) ====================
from src.prm.train_precog import PrecogProtoNet as AblatedProtoNet
import torch

# ==================== eval (identical numerics to the notebook) ====================
from sklearn.metrics import roc_auc_score

def support_draws(t, k=KSUP_EVAL, n_repeats=CMP_REPEATS, seed=SEED):
    rng = np.random.RandomState(seed)
    y = np.array([t["y"][r] for r in t["rids"]])
    pi0, fi0 = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n_repeats):
        pi, fi = rng.permutation(pi0), rng.permutation(fi0)
        sup, qry = np.concatenate([pi[:k], fi[:k]]), np.concatenate([pi[k:], fi[k:]])
        if len(set(y[sup])) < 2 or len(set(y[qry])) < 2: continue
        out.append((sup, qry))
    return out

def _encode_all(prm, t, batch=0):
    rids = t["rids"]
    if not batch:
        return np.stack([prm._encode_rollout(t["aseq"][int(r)], t["rseq"][int(r)]) for r in rids])
    pairs = [prm._interleave_rollout(t["aseq"][int(r)], t["rseq"][int(r)]) for r in rids]
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch):
            toks = [p[0] for p in pairs[i:i + batch]]
            poss = [p[1] for p in pairs[i:i + batch]]
            out.append(prm._encode_batch(toks, poss).cpu().numpy())
    return np.concatenate(out)

def eval_protonet_task(prm, t, k=KSUP_EVAL, batch=0):
    E = _encode_all(prm, t, batch=batch)
    y = np.array([t["y"][r] for r in t["rids"]])
    aucs = []
    for sup, qry in support_draws(t, k):
        cp, cf = E[sup][y[sup] == 1].mean(0), E[sup][y[sup] == 0].mean(0)
        sc = np.linalg.norm(E[qry] - cf, axis=1) - np.linalg.norm(E[qry] - cp, axis=1)
        aucs.append(roc_auc_score(y[qry], sc))
    return float(np.mean(aucs)) if aucs else np.nan

def eval_on_cells(prm, cell_list, test_labels, batch=0):
    out = {}
    for ds, m in cell_list:
        per = {}
        for t in eval_tasks_for(ds, m, test_labels):
            a = eval_protonet_task(prm, t, batch=batch)
            if not np.isnan(a): per[t["label"]] = a
        out[cell_key(ds, m)] = per
        if per: print(f"    {ds}/{m}: {len(per)} tasks  AUROC={np.mean(list(per.values())):.3f}",
                      flush=True)
    return out

# ==================== training + part files ====================

def _rid_subset(t, cap, salt):
    rids = list(t["rids"])
    if cap is None or len(rids) <= cap: return rids
    rng = np.random.RandomState(zlib.crc32(salt.encode()) % (2**31))
    rng.shuffle(rids)
    return rids[:cap]

def _gather(cell_list, test_labels, max_per_task=None):
    act, res, lab, tid, mids, dsids = [], [], [], [], [], []
    for ds, m in cell_list:
        for t in train_tasks_for(ds, m, test_labels):
            for r in _rid_subset(t, max_per_task, f"{ds}|{m}|{t['label']}"):
                r = int(r)
                act.append(t["aseq"][r]); res.append(t["rseq"][r]); lab.append(t["y"][r])
                tid.append(f"{ds}|{m}|{t['label']}")
                mids.append(MODELS.index(m)); dsids.append(DATASETS.index(ds))
    print(f"  train pool: {len(set(tid))} (cell,task) groups, {len(lab)} rollouts", flush=True)
    return act, res, lab, tid, mids, dsids

def traj_key(pool, ablation, seed): return f"{pool}|{ablation}|s{seed}"

def part_path(parts_dir, key):
    return parts_dir / (key.replace("|", "__").replace(":", "_").replace("+", "-") + ".json")

def _write_part(parts_dir, key, pool, ablation, seed, epochs_dict):
    parts_dir.mkdir(parents=True, exist_ok=True)
    p = part_path(parts_dir, key)
    tmp = p.with_suffix(".tmp")
    json.dump({"key": key, "pool": pool, "ablation": ablation, "seed": seed,
               "epochs": epochs_dict}, open(tmp, "w"))
    os.replace(tmp, p)

def completed_epochs(key, parts_dir, traj_path):
    eps = set()
    if traj_path.exists():
        eps |= set(json.load(open(traj_path)).get(key, {}))
    p = part_path(parts_dir, key)
    if p.exists():
        eps |= set(json.load(open(p)).get("epochs", {}))
    return eps

def run_pool(pool, args, test_labels):
    key = traj_key(pool, args.ablation, args.seed)
    if str(args.epochs) in completed_epochs(key, args.parts_dir, args.traj_path) and not args.force:
        print(f"[skip] {key} already complete", flush=True)
        return
    spec = ABLATIONS[args.ablation]
    act, res, lab, tid, mids, dsids = _gather(POOLS[pool], test_labels, max_per_task=args.cap)
    kw = {**PROTO_KWARGS, **spec.get("proto", {}),
          "epochs": args.epochs, "episodes_per_epoch": args.episodes}
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    prm = AblatedProtoNet(**kw)
    prm.input_mode = spec.get("input_mode", "both")
    if "pooling" in spec: prm.pooling = spec["pooling"]
    for a, v in spec.get("attrs", {}).items(): setattr(prm, a, v)
    ids = {"model": mids, "dataset": dsids}.get(spec.get("adv"))
    epochs_dict, eval_cells = {}, [c for c in cells_grid()
                                   if not args.eval_cells or cell_key(*c) in args.eval_cells]

    def _cb(prm_, epoch, loss):
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"  -- {key}  epoch {epoch}  (loss={loss:.4f}) --", flush=True)
            epochs_dict[str(epoch)] = eval_on_cells(prm_, eval_cells, test_labels,
                                                    batch=args.batched_eval)
            _write_part(args.parts_dir, key, pool, args.ablation, args.seed, epochs_dict)

    t0 = time.time()
    prm.fit_from_embeddings(act, res, lab, tid, model_ids=ids, epoch_callback=_cb)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    prm.save(str(args.ckpt_dir /
                 f"pool_{pool.replace(':', '_').replace('+', '-')}_{args.ablation}_s{args.seed}.pt"))
    print(f"[done] {key} in {(time.time() - t0) / 60:.1f} min", flush=True)

# ==================== merge / status / plan ====================

def merge(args):
    traj = json.load(open(args.traj_path)) if args.traj_path.exists() else {}
    n = 0
    for p in sorted(args.parts_dir.glob("*.json")):
        part = json.load(open(p))
        cur = traj.setdefault(part["key"], {})
        add = {ep: v for ep, v in part["epochs"].items() if ep not in cur}
        cur.update(add)
        if add: n += 1
        print(f"  {part['key']}: epochs {sorted(part['epochs'], key=int)} "
              f"({'merged ' + str(len(add)) + ' new' if add else 'already present'})")
    json.dump(traj, open(args.traj_path, "w"))
    print(f"merged {n} part files into {args.traj_path} "
          f"(re-run the notebook's trajectory-store cell to reload TRAJ)")

def status(args):
    print(f"{'pool':<32}{'cells':>6}  status")
    for pool in POOLS:
        key = traj_key(pool, args.ablation, args.seed)
        eps = completed_epochs(key, args.parts_dir, args.traj_path)
        st = ("done" if str(args.epochs) in eps
              else f"partial({','.join(sorted(eps, key=int))})" if eps else "-")
        print(f"{pool:<32}{len(POOLS[pool]):>6}  {st}")

def plan(args):
    remaining = [p for p in POOLS
                 if str(args.epochs) not in completed_epochs(
                     traj_key(p, args.ablation, args.seed), args.parts_dir, args.traj_path)]
    remaining.sort(key=lambda p: -len(POOLS[p]))
    buckets = [[] for _ in range(args.plan)]
    for i, p in enumerate(remaining):
        buckets[i % args.plan].append(p)
    print(f"# {len(remaining)} pools remaining -> {args.plan} workers "
          f"(run each command in its own terminal):")
    for b in buckets:
        if b:
            flags = f" --epochs {args.epochs} --eval-every {args.eval_every} " \
                    f"--episodes {args.episodes} --cap {'none' if args.cap is None else args.cap}"
            print(f"python scripts/pool_traj_worker.py{flags} " + " ".join(f"'{p}'" for p in b))

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*", help="pool names to run sequentially (see --list)")
    ap.add_argument("--convert", action="store_true", help="build mmap caches from slim pickles")
    ap.add_argument("--merge", action="store_true", help="fold part files into trajectories.json")
    ap.add_argument("--list", action="store_true", help="show pool status")
    ap.add_argument("--plan", type=int, help="split remaining pools into N worker commands")
    ap.add_argument("--ablation", default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20, help="MUST match the notebook's TRAJ_EPOCHS")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--cap", default="30", help="rollouts/task cap ('none' to disable); "
                                                "MUST match the notebook's TRAJ_CAP")
    ap.add_argument("--batched-eval", type=int, default=0, metavar="B",
                    help="encode eval rollouts in batches of B (0 = per-rollout, as notebook)")
    ap.add_argument("--eval-cells", default="", help="debug: comma list of cells to eval "
                                                     "(default all 9)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--parts-dir", type=Path, default=ABL_DIR / "traj_parts")
    ap.add_argument("--ckpt-dir", type=Path, default=ABL_DIR / "ckpts")
    ap.add_argument("--traj-path", type=Path, default=ABL_DIR / "trajectories.json")
    args = ap.parse_args()
    args.cap = None if str(args.cap).lower() == "none" else int(args.cap)
    args.eval_cells = {c.strip() for c in args.eval_cells.split(",") if c.strip()}

    if args.convert:
        for ds, m in cells_grid():
            convert_cell(ds, m)
        return
    if args.merge:
        merge(args); return
    if args.list:
        status(args); return
    if args.plan:
        plan(args); return
    if not args.pools:
        ap.print_help(); return

    bad = [p for p in args.pools if p not in POOLS]
    if bad:
        sys.exit(f"unknown pools: {bad}\nvalid: {', '.join(POOLS)}")
    if not SPLITS_PATH.exists():
        sys.exit(f"missing {SPLITS_PATH} (run the notebook's split cell first)")
    test_labels = {ds: set(v) for ds, v in json.load(open(SPLITS_PATH)).items()}
    print(f"config: epochs={args.epochs} eval_every={args.eval_every} episodes={args.episodes} "
          f"cap={args.cap} ablation={args.ablation} seed={args.seed} "
          f"batched_eval={args.batched_eval}", flush=True)
    load_all_cells()
    for pool in args.pools:
        run_pool(pool, args, test_labels)

if __name__ == "__main__":
    main()
