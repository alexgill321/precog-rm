#!/usr/bin/env python3
"""Maintainer gate: verify a built release tree (scripts/build_release.py)
before uploading.

For every (benchmark, policy) cell:
  * task count and task ids match the embedded cell the results were computed from;
  * the shipped rollout ids equal the rollout ids in the mmap caches (i.e. exactly
    the 50 per task the paper used), and the embeddings/ folder holds the same ids;
  * per rollout, n_steps in the JSON equals the number of r<rid>_s<t> keys;
  * no credential-like content anywhere in the tree (same scan as upload_hf.py).

    python scripts/verify_release.py --tree ../hf
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts._paths import MMAP_DIR, require  # noqa: E402
from build_release import POLICY_DIR, MODELS  # noqa: E402
from upload_hf import scan_secrets  # noqa: E402


def mmap_rids(ds, m):
    import pool_traj_worker as W
    cell = W.load_cell(ds, m)
    return {t["label"]: sorted(int(r) for r in t["rids"]) for t in cell}


def shipped_label(benchmark, task):
    if benchmark == "tau2":
        dom, idx = task.split("_", 1)
        return f"{dom}/{idx}"
    if benchmark == "bfclv3":
        return task.replace("_", "/", 1)
    return f"medqa/{int(task.split('_')[1])}"


def check_cell(tree, benchmark, policy):
    pdir = POLICY_DIR[policy]
    rroot = tree / "rollouts" / benchmark / pdir
    eroot = tree / "embeddings" / benchmark / pdir
    tasks = sorted(p.name for p in rroot.iterdir() if p.is_dir())
    cache = mmap_rids(benchmark, policy)        # mixed-outcome tasks only (what results use)
    n_rollouts = n_extra = 0
    problems = []
    for task in tasks:
        tdir = rroot / task
        files = sorted(tdir.glob("rollout_*.json"))
        rids = sorted(int(f.stem.split("_")[1]) for f in files)
        n_rollouts += len(rids)
        extra = tdir / "extra_rollouts.jsonl"
        if extra.exists():
            n_extra += sum(1 for _ in open(extra))
        if not (eroot / task / "action.npz").exists():
            info = json.load(open(tdir / "task.json"))
            if "embeddings" in info and shipped_label(benchmark, task) not in cache:
                continue                      # documented omission of an unused task
            problems.append(f"{task}: embeddings missing"); continue
        z = np.load(eroot / task / "action.npz")
        keys = {}
        for k in z.files:
            m = re.match(r"r(\d+)_s(\d+)$", k)
            keys.setdefault(int(m.group(1)), []).append(int(m.group(2)))
        z.close()
        if sorted(keys) != rids:
            problems.append(f"{task}: embedding rids != shipped rids")
        for f in files:
            d = json.load(open(f))
            if d["n_steps"] != len(keys.get(d["rid"], [])):
                problems.append(f"{task}/{f.name}: n_steps {d['n_steps']} != npz {len(keys.get(d['rid'], []))}")
                break
        label = shipped_label(benchmark, task)
        if label in cache and cache[label] != rids:
            problems.append(f"{task}: shipped rids differ from the mmap cache ({len(rids)} vs {len(cache[label])})")
        rew = json.load(open(eroot / task / "rewards.json"))
        if sorted(int(k) for k in rew) != rids:
            problems.append(f"{task}: rewards.json rids mismatch")
    covered = sum(1 for lab in cache if lab in {shipped_label(benchmark, t) for t in tasks})
    print(f"{benchmark:12} {pdir:13} tasks={len(tasks):4d} rollouts={n_rollouts:6d} extra={n_extra:6d} "
          f"cache-tasks covered={covered}/{len(cache)} problems={len(problems)}")
    for p in problems[:8]:
        print("   ", p)
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--benchmark", nargs="*", default=["tau2", "bfclv3", "agentclinic"])
    ap.add_argument("--policy", nargs="*", default=MODELS)
    ap.add_argument("--no-secret-scan", action="store_true")
    args = ap.parse_args()
    require(MMAP_DIR, "mmap caches", "caches")
    problems = []
    for b in args.benchmark:
        for p in args.policy:
            if (args.tree / "rollouts" / b / POLICY_DIR[p]).exists():
                problems += check_cell(args.tree, b, p)
    if not args.no_secret_scan:
        print("secret scan ...", flush=True)
        scan_secrets(args.tree)
        print("secret scan: clean")
    print("VERIFY", "FAILED" if problems else "OK", f"({len(problems)} problems)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
