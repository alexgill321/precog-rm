#!/usr/bin/env python3
"""Download the data release from HuggingFace into $PRECOG_DATA (default <repo>/data)
and lay it out the way the code expects.

The release is one dataset repo (agill32/precog-rm) with four browsable parts,
plus a model repo for checkpoints:

  rollouts/<benchmark>/<policy>/<task>/rollout_NNN.json   raw trajectories (+ extra_rollouts.jsonl, task.json)
  embeddings/<benchmark>/<policy>/<task>/{action,observation}.npz + rewards.json
  grpo/{runs,replay,eval}/                                  the 21 official GRPO runs and replay results
  caches/mmap/                                              optional float16 fast path
  ablations/agentclinic/{notools,resample,terra}, meta/     AgentClinic ablation data, task lists
  checkpoints (model repo agill32/precog-rm-checkpoints)    -> ckpts/

After downloading, `materialize()` builds `table1_data/<benchmark>_<model>/`
(reward stubs + embeddings/{action,response}/<task>.npz as hard links) so every
script runs unchanged, and extracts any tarball.

    python scripts/download.py --part checkpoints caches            # enough to re-evaluate everything
    python scripts/download.py --part embeddings --benchmark tau2   # one benchmark's embeddings
    python scripts/download.py --part rollouts --benchmark agentclinic --policy qwen3-80b
    python scripts/download.py --all
"""
import argparse
import json
import os
import re
from pathlib import Path

from _paths import DATA, POLICY_DIR, DATASETS, MODELS

ORG = "agill32"
DATASET_REPO = f"{ORG}/precog-rm"
MODEL_REPO = f"{ORG}/precog-rm-checkpoints"
PARTS = ["rollouts", "embeddings", "grpo", "caches", "ablations", "meta", "checkpoints"]
STEM = {"tau2": lambda t: t, "bfclv3": lambda t: t, "agentclinic": lambda t: "medqa_" + str(int(t.split("_")[1]))}
CAT = {"tau2": lambda t: t.split("_", 1), "bfclv3": lambda t: t.split("_", 1),
       "agentclinic": lambda t: ("medqa", str(int(t.split("_")[1])))}


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src.resolve())


def materialize(root: Path):
    """embeddings/<b>/<policy>/<task>/ -> table1_data/<b>_<model>/ (what the loaders read)."""
    inv = {v: k for k, v in POLICY_DIR.items()}
    n = 0
    for b in DATASETS:
        for pdir, model in inv.items():
            src = root / "embeddings" / b / pdir
            if not src.exists():
                continue
            cell = root / "table1_data" / f"{b}_{model}"
            for tdir in sorted(p for p in src.iterdir() if p.is_dir()):
                task = tdir.name
                stem = STEM[b](task)
                link(tdir / "action.npz", cell / "embeddings" / "action" / f"{stem}.npz")
                link(tdir / "observation.npz", cell / "embeddings" / "response" / f"{stem}.npz")
                cat, tid = CAT[b](task)
                rewards = json.load(open(tdir / "rewards.json"))
                for rid, rew in rewards.items():
                    stub = cell / cat / f"{tid}_tdp_{int(rid):02d}.json"
                    if not stub.exists():
                        stub.parent.mkdir(parents=True, exist_ok=True)
                        json.dump({"reward": rew}, open(stub, "w"))
                n += 1
            # tau2/qwen: the loaders' permanent-support and replay code wants the consolidated manifest
            man = root / "rollouts" / b / pdir / "manifest.json"
            if man.exists():
                link(man, cell / "manifest.json")
    if n:
        print(f"materialized {n} task cells under {root / 'table1_data'}")
    extract_tarballs(root)


def extract_tarballs(root: Path):
    import tarfile
    for tar in sorted(root.rglob("*.tar.gz")):
        marker = tar.with_suffix("").with_suffix(".extracted")
        if marker.exists():
            continue
        print(f"  extracting {tar.relative_to(root)}", flush=True)
        with tarfile.open(tar) as tf:
            tf.extractall(tar.parent, filter="data")
        marker.write_text("ok\n")
        tar.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", nargs="*", choices=PARTS, default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--benchmark", nargs="*", choices=DATASETS, help="filter rollouts/embeddings")
    ap.add_argument("--policy", nargs="*", choices=list(POLICY_DIR.values()), help="filter rollouts/embeddings")
    ap.add_argument("--materialize-only", action="store_true", help="skip downloading; rebuild table1_data from what is on disk")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.materialize_only:
        materialize(DATA); return
    from huggingface_hub import snapshot_download
    parts = PARTS if args.all else args.part
    if not parts:
        ap.error("give --part ... or --all")
    bs = args.benchmark or DATASETS
    ps = args.policy or list(POLICY_DIR.values())
    pats = []
    for p in parts:
        if p in ("rollouts", "embeddings"):
            pats += [f"{p}/{b}/{pol}/**" for b in bs for pol in ps]
            if p == "embeddings":
                pats += ["embeddings/reasoning/**", "embeddings/grpo_replay/**"]
        elif p != "checkpoints":
            pats.append(f"{p}/**")
    if pats:
        print(f"[{DATASET_REPO}] -> {DATA}: {len(pats)} patterns", flush=True)
        snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(DATA), allow_patterns=pats + ["README.md"])
    if "checkpoints" in parts:
        print(f"[{MODEL_REPO}] -> {DATA / 'ckpts'}", flush=True)
        snapshot_download(repo_id=MODEL_REPO, repo_type="model", local_dir=str(DATA / "ckpts"))
    materialize(DATA)
    print("done:", DATA)


if __name__ == "__main__":
    main()
