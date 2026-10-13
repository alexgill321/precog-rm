#!/usr/bin/env python3
"""Move the runner's flat staging output into the released tau2 cell layout.

The runner writes `<staging>/<domain>/<task>_tdp_<NN>.json`. The release layout
(what every loader reads) is `<cell>/<domain>/<idx>/rollout_<NNN>.json`, where
`idx` is the task's position-stable index: airline/retail tasks keep their
numeric id, telecom tasks (long descriptive ids) are mapped through
meta/tau2_telecom_id_mapping.json. Also works in reverse (--seed) to
pre-populate a staging dir from an existing cell so the runner resumes.

    python -m generation.tau2.reorganize_staging --staging rollouts/tau2/_staging --cell $PRECOG_DATA/table1_data/tau2_deepseek
    python -m generation.tau2.reorganize_staging --staging ... --cell ... --seed      # cell -> staging (resume)
"""
import argparse
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PRECOG_DATA", REPO / "data"))
TELECOM_MAP = DATA / "meta" / "tau2_telecom_id_mapping.json"


def mappings():
    """(original task id -> idx) and inverse, per domain."""
    tel = json.load(open(TELECOM_MAP)) if TELECOM_MAP.exists() else {}
    orig_to_idx = {k.split("/", 1)[1]: v.split("/", 1)[1] for k, v in tel.items()}
    idx_to_orig = {v: k for k, v in orig_to_idx.items()}
    return orig_to_idx, idx_to_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--seed", action="store_true", help="copy cell -> staging instead of staging -> cell")
    ap.add_argument("--copy", action="store_true", help="copy instead of move (staging -> cell)")
    args = ap.parse_args()
    orig_to_idx, idx_to_orig = mappings()
    n = 0
    if args.seed:
        for dom_dir in sorted(p for p in args.cell.iterdir() if p.is_dir() and p.name != "embeddings"):
            for task_dir in sorted(p for p in dom_dir.iterdir() if p.is_dir()):
                orig = idx_to_orig.get(task_dir.name, task_dir.name) if dom_dir.name == "telecom" else task_dir.name
                (args.staging / dom_dir.name).mkdir(parents=True, exist_ok=True)
                for rf in task_dir.glob("rollout_*.json"):
                    rid = int(rf.stem.split("_")[1])
                    dest = args.staging / dom_dir.name / f"{orig}_tdp_{rid:02d}.json"
                    if not dest.exists():
                        shutil.copy2(rf, dest); n += 1
        print(f"seeded {n} rollouts into {args.staging}")
        return
    for dom_dir in sorted(p for p in args.staging.iterdir() if p.is_dir()):
        for f in sorted(dom_dir.glob("*_tdp_*.json")):
            orig, rid = f.stem.rsplit("_tdp_", 1)
            idx = orig_to_idx.get(orig, orig) if dom_dir.name == "telecom" else orig
            dest_dir = args.cell / dom_dir.name / idx
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"rollout_{int(rid):03d}.json"
            if dest.exists():
                continue
            (shutil.copy2 if args.copy else shutil.move)(str(f), dest); n += 1
    print(f"placed {n} rollouts into {args.cell}")


if __name__ == "__main__":
    main()
