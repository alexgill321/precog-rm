#!/usr/bin/env python3
"""Build a nodiag table1_data dir from an existing v2 one WITHOUT re-embedding.

The nodiag leakage ablation asks: how much AgentClinic separability comes from
the discriminator reading the final DIAGNOSIS action, versus the trajectory
prefix? We answer it by dropping each rollout's terminal diagnosis STEP from the
already-computed v2 embeddings, then re-running clustering/discriminators on the
trimmed sequences.

We drop the final step ONLY for rollouts that actually ended in a diagnosis
(others — e.g. those that ran out of inferences — have no diagnosis to leak and
are left intact). Reward stubs are copied verbatim.

Usage:
    python scripts/make_agentclinic_nodiag.py --source qwen_medqa
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

import os
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DATA = Path(os.environ.get("PRECOG_DATA", REPO / "data"))

from generation.agentclinic.embed_agentclinic import SOURCES, _ROLLOUT_RE, _extract_action_response  # noqa: E402

_KEY_RE = re.compile(r"r(\d+)_s(\d+)")


def ends_in_diagnosis(history) -> bool:
    actions, _ = _extract_action_response(history, action_mode="action")
    return bool(actions) and actions[-1].startswith("DIAGNOSIS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES),
                    help="e.g. qwen_medqa (uses table1_data_<source>_v2)")
    args = ap.parse_args()

    rollouts_dir = SOURCES[args.source]["rollouts_dir"]
    tag = f"{args.source}_v2"
    src_dir = DATA / "table1_data" / f"agentclinic_{args.source.split('_')[0]}"
    dst_dir = DATA / "table1_data" / f"agentclinic_{args.source.split('_')[0]}_nodiag"
    domain = "medqa"

    if not src_dir.exists():
        raise SystemExit(f"missing {src_dir} — repack the v2 embeddings first")

    # Per-(sid, rid) flag: did this rollout end in a diagnosis?
    diag: dict[tuple[int, int], bool] = {}
    for fp in rollouts_dir.glob("scenario_*_rollout_*.json"):
        m = _ROLLOUT_RE.search(fp.name)
        if not m:
            continue
        sid, rid = int(m.group(1)), int(m.group(2))
        diag[(sid, rid)] = ends_in_diagnosis(json.load(open(fp)).get("history", []))

    # Reward stubs are unchanged by the ablation.
    (dst_dir / domain).mkdir(parents=True, exist_ok=True)
    for j in (src_dir / domain).glob("*.json"):
        shutil.copyfile(j, dst_dir / domain / j.name)

    for sub in ("action", "response"):
        (dst_dir / "embeddings" / sub).mkdir(parents=True, exist_ok=True)

    n_dropped = n_keys = n_emptied = 0
    for npz_path in sorted((src_dir / "embeddings" / "action").glob(f"{domain}_*.npz")):
        sid = int(npz_path.stem.split("_")[1])
        # steps present per rid (same across action/response by construction)
        with np.load(npz_path) as z:
            by_rid: dict[int, list[int]] = {}
            for k in z.files:
                mm = _KEY_RE.match(k)
                by_rid.setdefault(int(mm.group(1)), []).append(int(mm.group(2)))
        drop = {f"r{rid}_s{max(steps)}"
                for rid, steps in by_rid.items()
                if diag.get((sid, rid), False)}
        n_emptied += sum(1 for rid, steps in by_rid.items()
                         if diag.get((sid, rid), False) and len(steps) == 1)

        for sub in ("action", "response"):
            with np.load(src_dir / "embeddings" / sub / f"{domain}_{sid}.npz") as z:
                out = {k: z[k] for k in z.files if k not in drop}
                if sub == "action":
                    n_dropped += len(drop)
                    n_keys += len(z.files)
            np.savez_compressed(
                dst_dir / "embeddings" / sub / f"{domain}_{sid}.npz", **out)

    print(f"{tag}_nodiag: dropped {n_dropped} diagnosis steps of {n_keys} "
          f"action keys ({n_dropped / max(n_keys,1):.1%}); "
          f"{n_emptied} rollouts became empty -> {dst_dir}")


if __name__ == "__main__":
    main()
