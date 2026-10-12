#!/usr/bin/env python3
"""Appendix B.1 (tab:reasoning): does the agent's internal reasoning r_t add
signal over the action--observation stream?

Tau2-Bench / Qwen3-Next-80B-Thinking (the only policy exposing its reasoning),
held-out test tasks of the Table 1 evaluation set. Three channels under the
exact channel-ablation protocol (3-fold CV, fold-level aggregation):
  AO     the paired construction (reference)
  R      per-step reasoning embeddings alone
  AO+R   the concatenated [a_t || o_t || r_t] step vector
Raw ceiling = best of NC / RBF on the channel's pooled vector; clustered ceiling
= best fastText n in {1,2,5} on HDBSCAN states clustered from the channel's own
step embeddings.

Step-aligned reasoning embeddings (text-embedding-3-large of each step's
reasoning_content, "<empty>" placeholder where absent) are shipped under
$PRECOG_DATA/reasoning_emb/tau2_qwen_steps/<task>.npz (keys r<rid>). They can
be regenerated from the raw tau2/qwen rollouts with --embed (needs
OPENAI_API_KEY, ~$6).

    python scripts/reasoning_signal_ablation.py
    python scripts/reasoning_signal_ablation.py --embed    # rebuild the embedding cache first
"""
import argparse
import json
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from _paths import ABL, REASONING_EMB, MMAP_DIR, SPLITS, TABLE1, require
from q4_channel_ablation import fold_aurocs, aggregate, _cluster, _unit

warnings.filterwarnings("ignore"); np.seterr(all="ignore")
STEPS = REASONING_EMB / "tau2_qwen_steps"
OUT = ABL / "reasoning_ablation.json"


def embed_cache(max_rollouts=50, concurrency=8):
    """Rebuild the per-step reasoning cache from the raw tau2/qwen rollouts (OpenAI API)."""
    import asyncio
    from openai import AsyncOpenAI
    data = require(TABLE1 / "tau2_qwen", "tau2/qwen rollouts", "embeddings")
    test = set(json.load(open(SPLITS))["tau2"])
    client, sem = AsyncOpenAI(timeout=120.0), asyncio.Semaphore(concurrency)

    async def one(label, td):
        pk = STEPS / (label.replace("/", "_") + ".npz")
        if pk.exists():
            return
        out = {}
        for rf in sorted(td.glob("rollout_*.json"))[:max_rollouts]:
            r = json.load(open(rf))
            texts = []
            for s in r["steps"]:
                smp = next((x for x in (s.get("samples") or []) if x.get("is_chosen")), None)
                texts.append(((smp or {}).get("reasoning_content") or "<empty>")[:24000])
            embs = []
            for i in range(0, len(texts), 64):
                async with sem:
                    resp = await client.embeddings.create(model="text-embedding-3-large", input=texts[i:i + 64])
                embs.extend(d.embedding for d in resp.data)
            out[int(rf.stem.split("_")[1])] = np.asarray(embs, dtype=np.float32)
        STEPS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pk, **{f"r{k}": v for k, v in out.items()})
        print(f"  embedded {label}: {len(out)} rollouts", flush=True)

    async def run():
        for dom in ("airline", "retail", "telecom"):
            for td in sorted((data / dom).iterdir()):
                if td.is_dir() and f"{dom}/{td.name}" in test:
                    await one(f"{dom}/{td.name}", td)
    asyncio.run(run())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--print-only", action="store_true", help="print the table from reasoning_ablation.json")
    args = ap.parse_args()
    if args.print_only:
        res = json.load(open(OUT))
        print(f"{'channel':6} {'N':>3} {'raw ceiling':>11} {'clustered ceiling':>17}")
        for name, r in res.items():
            a = r["agg"]
            print(f"{name:6} {r['n']:3d} {max(a['NC'][0], a['RBF'][0]):11.3f} "
                  f"{max(a[f'FT{n}'][0] for n in (1, 2, 5)):17.3f}")
        return
    if args.embed:
        embed_cache()
    require(STEPS, "per-step reasoning embeddings", "embeddings")
    require(MMAP_DIR, "mmap caches", "caches")
    import pool_traj_worker as W
    W.load_all_cells()
    test = {ds: set(v) for ds, v in json.load(open(SPLITS)).items()}

    tasks, truncated, skipped = [], 0, 0
    for t in W.eval_tasks_for("tau2", "qwen", test):
        pk = STEPS / (t["label"].replace("/", "_") + ".npz")
        if not pk.exists():
            continue
        z = np.load(pk); R = {int(k[1:]): z[k] for k in z.files}
        q, rids = {}, []
        for r in t["rids"]:
            r = int(r); seq = R.get(r)
            if seq is None:
                skipped += 1; continue
            L = len(t["aseq"][r])
            if len(seq) != L:          # observed: one trailing extra reasoning step
                truncated += 1; seq = seq[:L]
            q[r] = seq.astype(np.float32); rids.append(r)
        if len(rids) < 6:
            continue
        v = dict(t); v["rids"] = np.array(rids); v["qseq"] = q
        tasks.append(v)
    print(f"tau2/qwen: {len(tasks)} test tasks ({truncated} rollouts prefix-truncated, {skipped} skipped)")

    res = {}
    for name, chan in (("AO", "paired"), ("R", "R"), ("AO+R", "aor")):
        per = {}
        for t in tasks:
            rids = [int(r) for r in t["rids"]]
            if chan == "paired":
                vec = {r: np.asarray(t["pooled"][r], dtype=np.float64) for r in rids}
                paths, K = {r: list(t["path"][r]) for r in rids}, t["K"]
            else:
                seqs = {r: (t["qseq"][r] if chan == "R" else
                            np.concatenate([np.asarray(t["aseq"][r], dtype=np.float32),
                                            np.asarray(t["rseq"][r], dtype=np.float32), t["qseq"][r]], 1))
                        for r in rids}
                order = [(r, i) for r in rids for i in range(len(seqs[r]))]
                labels, K = _cluster(np.stack([seqs[r][i] for r, i in order]))
                paths, off = {}, 0
                for r in rids:
                    L = len(seqs[r]); paths[r] = [int(labels[off + i]) for i in range(L)]; off += L
                vec = {r: _unit(seqs[r].astype(np.float64).mean(0)) for r in rids}
            per[t["label"]] = fold_aurocs(t, vec, paths, K)
        agg = aggregate(per)
        res[name] = {"n": len(per), "agg": agg, "per_task": per}
        raw = max(agg["NC"][0], agg["RBF"][0]); clu = max(agg[f"FT{n}"][0] for n in (1, 2, 5))
        print(f"  {name:5} n={len(per)}  raw ceiling={raw:.3f} (NC {agg['NC'][0]:.3f}, RBF {agg['RBF'][0]:.3f})"
              f"  clustered ceiling={clu:.3f} (FT1 {agg['FT1'][0]:.3f}, FT2 {agg['FT2'][0]:.3f}, FT5 {agg['FT5'][0]:.3f})",
              flush=True)
    json.dump(res, open(OUT, "w"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
