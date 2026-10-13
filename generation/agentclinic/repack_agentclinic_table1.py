#!/usr/bin/env python3
"""Repack AgentClinic embeddings into table1_data layout.

Source: $PRECOG_DATA/agentclinic_embeddings/<tag>/
    manifest.json                      (scenario rewards)
    embeddings/{action,response}/scenario_XXXX.npz

Target: $PRECOG_DATA/table1_data/agentclinic_<model>/
    medqa/{id}_tdp_{rid:02d}.json      (reward stubs)
    embeddings/{action,response}/medqa_{id}.npz

Usage:
    python scripts/repack_agentclinic_table1.py --tag gpt5mini_medqa
    python scripts/repack_agentclinic_table1.py --tag deepseek_medqa
"""
import argparse
import json
import shutil
from pathlib import Path

import os
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PRECOG_DATA", REPO / "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="dir name under exp/agentclinic/embeddings/")
    args = ap.parse_args()

    src = DATA / "agentclinic_embeddings" / args.tag
    dst = DATA / "table1_data" / f"agentclinic_{args.tag.split('_')[0]}"

    with open(src / "manifest.json") as f:
        manifest = json.load(f)
    domain = manifest.get("dataset", "medqa").split("_")[0].lower()

    n_tasks = 0
    for sc in manifest["scenarios"]:
        sid = sc["scenario_id"]
        stem = sc["stem"]
        rewards = sc["rewards"]

        act_src = src / "embeddings/action" / f"{stem}.npz"
        res_src = src / "embeddings/response" / f"{stem}.npz"
        if not act_src.exists() or not res_src.exists():
            print(f"SKIP scenario {sid}: missing NPZ")
            continue

        task_dir = dst / domain
        task_dir.mkdir(parents=True, exist_ok=True)
        rids = sorted(int(k) for k in rewards.keys())
        assert rids == list(range(len(rids))), f"non-dense rids for {stem}"
        for rid in rids:
            with open(task_dir / f"{sid}_tdp_{rid:02d}.json", "w") as f:
                json.dump({"reward": rewards[str(rid)]}, f)

        for sub, s in [("action", act_src), ("response", res_src)]:
            emb_dir = dst / "embeddings" / sub
            emb_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, emb_dir / f"{domain}_{sid}.npz")

        n_tasks += 1

    print(f"Repacked {n_tasks} tasks into {dst}")


if __name__ == "__main__":
    main()
