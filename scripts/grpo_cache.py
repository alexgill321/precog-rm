#!/usr/bin/env python3
"""Build the per-rollout token cache from the raw GRPO rollout logs.

Streams every $PRECOG_DATA/runs/<run>/rollouts/batch_*.jsonl once and writes
one parquet per run to notebooks/data/tau2_es/rollout_cache/<variant>_s<seed>.parquet
with the compact per-rollout record the GRPO analysis reads (assigned/original
reward, early-stop info, PRM score sequence, exact prefill/sample token counts).
The shipped caches were built by this code; `--check` rebuilds into a temp
directory and compares token totals against the shipped files.

    python scripts/grpo_cache.py            # build missing caches
    python scripts/grpo_cache.py --rebuild  # force
    python scripts/grpo_cache.py --check    # verify shipped caches against raw logs
"""
import argparse
import json
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from grpo_common import VARIANTS, SEEDS, run_dir, cache_name, CACHE_DIR


def parse_batch_file(fp):
    batch = int(re.search(r"batch_(\d+)", str(fp)).group(1))
    rows, n_err = [], 0
    with open(fp, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "prefill_tokens" not in r:      # failed-rollout stub
                n_err += 1
                continue
            conv = r.get("conversation") or []
            samp_chars = 0
            for s in r.get("steps") or []:
                for smp in s.get("samples") or []:
                    if smp.get("is_chosen"):
                        c = smp.get("content")
                        if c:
                            samp_chars += len(c)
                        tc = smp.get("tool_calls")
                        if tc:
                            samp_chars += len(str(tc))
            prm = r.get("prm_scores") or []
            rows.append(dict(
                batch=batch, domain=r.get("domain", "retail"), task_id=str(r["task_id"]),
                rep=r.get("rep", -1), reward=float(r["reward"]),
                original_reward=float(r.get("original_reward", r["reward"])),
                early_stopped=bool(r.get("early_stopped", False)),
                early_stop_turn=r.get("early_stop_turn"), predicted_reward=r.get("predicted_reward"),
                num_turns=int(r.get("num_turns", 0)),
                prefill_tokens=int(r["prefill_tokens"]), sample_tokens=int(r["sample_tokens"]),
                duration=float(r.get("duration") or 0.0), termination_reason=r.get("termination_reason"),
                n_prm=len(prm), prm_first=prm[0] if prm else float("nan"),
                prm_last=prm[-1] if prm else float("nan"), prm_max=max(prm) if prm else float("nan"),
                prm_min=min(prm) if prm else float("nan"), prm_scores=prm,
                conv_chars=sum(len(m.get("content") or "") for m in conv), samp_chars=samp_chars,
            ))
    return rows, n_err


def build(variant, seed, out_dir, workers=8):
    files = sorted((run_dir(variant, seed) / "rollouts").glob("batch_*.jsonl"))
    rows, n_err = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, e in ex.map(parse_batch_file, [str(f) for f in files]):
            rows.extend(r); n_err += e
    df = pd.DataFrame(rows)
    df.insert(0, "run", cache_name(variant, seed))
    out = Path(out_dir) / f"{cache_name(variant, seed)}.parquet"
    df.to_parquet(out, index=False)
    print(f"  {cache_name(variant, seed)}: {len(files)} batch files -> {len(df)} rollouts "
          f"({n_err} failed stubs skipped)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if args.check:
        tmp = Path(tempfile.mkdtemp(prefix="grpo_cache_check_"))
        bad = 0
        for v in args.variants:
            for s in SEEDS:
                new = pd.read_parquet(build(v, s, tmp, args.workers))
                old = pd.read_parquet(CACHE_DIR / f"{cache_name(v, s)}.parquet")
                cols = ["batch", "task_id", "rep", "reward", "prefill_tokens", "sample_tokens", "early_stopped"]
                same = (len(new) == len(old)
                        and new[cols].sort_values(["batch", "task_id", "rep"]).reset_index(drop=True)
                               .equals(old[cols].sort_values(["batch", "task_id", "rep"]).reset_index(drop=True)))
                print(f"    {cache_name(v, s)}: {'match' if same else 'MISMATCH'}")
                bad += not same
        print("check:", "all caches match the raw logs" if not bad else f"{bad} mismatches")
        return
    for v in args.variants:
        for s in SEEDS:
            out = CACHE_DIR / f"{cache_name(v, s)}.parquet"
            if out.exists() and not args.rebuild:
                continue
            build(v, s, CACHE_DIR, args.workers)
    print("done:", CACHE_DIR)


if __name__ == "__main__":
    main()
