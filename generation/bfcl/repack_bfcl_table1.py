#!/usr/bin/env python3
"""Repack BFCL saturation embeddings (any version) into table1_data layout.

Source: <emb_dir>/ (output of embed_bfcl_percall.py / embed_bfcl_bundled.py)
    multi_turn_{cat}_{id}_action.npz / _response.npz / _rewards.json

Target: <dst>/ (a table1_data cell)
    {short}/{id}_tdp_{rid:02d}.json          (reward stubs)
    {short}/{id}_kinds.json                  (per-rollout step kinds, v3 runs only)
    embeddings/action/{short}_{id}.npz
    embeddings/response/{short}_{id}.npz

Step kinds ("tool_response" | "turn_end" | "decode_fail") are re-derived from
the raw run logs via parse_task_rollouts_v3 and only align with v3-format
embeddings (bundled steps). Pass --kinds only for v3 sources.

Usage:
    python scripts/repack_bfcl_table1.py --src embeddings_dsv32     --dst table1_data_dsv32_v1
    python scripts/repack_bfcl_table1.py --src embeddings_dsv32_v2  --dst table1_data_dsv32_v2
    python scripts/repack_bfcl_table1.py --src embeddings_dsv32_v3  --dst table1_data_dsv32 \
        --kinds --runs-root exp/bfcl/runs/dsv32_194_prompt
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CAT_MAP = {
    "multi_turn_base": "base",
    "multi_turn_long_context": "longctx",
    "multi_turn_miss_func": "missfunc",
    "multi_turn_miss_param": "missparam",
}

PAT = re.compile(
    r"^(multi_turn_(?:base|long_context|miss_func|miss_param))_(\d+)_rewards\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="embedding dir written by embed_bfcl_*.py (per-task *_action.npz/_response.npz/_rewards.json)")
    ap.add_argument("--dst", required=True, type=Path,
                    help="output cell dir in table1_data layout, e.g. $PRECOG_DATA/table1_data/bfclv3_qwen")
    ap.add_argument("--kinds", action="store_true",
                    help="emit {id}_kinds.json step-kind sidecars (v3 only)")
    ap.add_argument("--runs-root", default=None,
                    help="BFCL runs root for kinds extraction")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    runs_root = Path(args.runs_root) if args.runs_root else None
    if args.kinds and runs_root is None:
        ap.error("--kinds requires --runs-root")

    if args.kinds:
        from generation.bfcl.embed_bfcl_bundled import parse_task_rollouts_v3

    n_tasks = 0
    for rewards_file in sorted(src.glob("*_rewards.json")):
        m = PAT.match(rewards_file.name)
        if not m:
            print(f"SKIP unrecognized: {rewards_file.name}")
            continue
        cat_long, task_id = m.group(1), m.group(2)
        short = CAT_MAP[cat_long]

        act_src = src / f"{cat_long}_{task_id}_action.npz"
        res_src = src / f"{cat_long}_{task_id}_response.npz"
        if not act_src.exists() or not res_src.exists():
            print(f"SKIP {cat_long}_{task_id}: missing NPZ")
            continue

        with open(rewards_file) as f:
            rewards = json.load(f)

        task_dir = dst / short
        task_dir.mkdir(parents=True, exist_ok=True)
        rids = sorted(int(k) for k in rewards.keys())
        assert rids == list(range(len(rids))), \
            f"non-dense rids for {cat_long}_{task_id}"
        for rid in rids:
            stub = task_dir / f"{task_id}_tdp_{rid:02d}.json"
            with open(stub, "w") as f:
                json.dump({"reward": rewards[str(rid)]}, f)

        for sub, s in [("action", act_src), ("response", res_src)]:
            emb_dir = dst / "embeddings" / sub
            emb_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, emb_dir / f"{short}_{task_id}.npz")

        if args.kinds:
            out = parse_task_rollouts_v3(
                runs_root / cat_long / f"{cat_long}_{task_id}",
                f"{cat_long}_{task_id}")
            kinds_by_rid: dict[str, list[str]] = {}
            for (rid, s), kind in zip(out["indices"], out["step_kinds"]):
                kinds_by_rid.setdefault(str(rid), []).append(kind)
            with open(task_dir / f"{task_id}_kinds.json", "w") as f:
                json.dump(kinds_by_rid, f)

        n_tasks += 1

    print(f"Repacked {n_tasks} tasks into {dst}"
          + (" (with step-kind sidecars)" if args.kinds else ""))


if __name__ == "__main__":
    main()
