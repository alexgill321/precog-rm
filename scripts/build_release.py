#!/usr/bin/env python3
"""Maintainer tool: build the browsable HuggingFace release tree from the
verified local sources (docs/HF_LAYOUT_PLAN.md).

    <out>/rollouts/<benchmark>/<policy>/<task>/rollout_NNN.json   the 50 rollouts the paper used
    <out>/rollouts/<benchmark>/<policy>/<task>/extra_rollouts.jsonl  surplus rollouts (tau2/qwen only)
    <out>/rollouts/<benchmark>/<policy>/<task>/task.json
    <out>/embeddings/<benchmark>/<policy>/<task>/{action.npz, observation.npz, rewards.json}

Each rollout JSON carries a common envelope: `steps` is the list of
(action, observation) texts in exactly the order the embeddings were built
(step t <-> key r<rid>_s<t>), `raw` is the untouched harness record. The
50-of-many rule is the loaders' own seeded subsample
(np.random.RandomState(42).choice(sorted(rids), 50)).

Inputs: the `table1_data/<benchmark>_<model>` cells and `rollouts/raw/` of the
staging data root ($PRECOG_DATA). Hard-links npz files when unchanged.

    python scripts/build_release.py --benchmark tau2 --out ../hf
    python scripts/build_release.py --benchmark bfclv3 agentclinic --out ../hf
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts._paths import DATA, TABLE1, ROLLOUTS  # noqa: E402

POLICY_DIR = {"gpt5mini": "gpt5mini", "qwen": "qwen3-80b", "deepseek": "deepseek-v3.2"}
MODELS = ["gpt5mini", "qwen", "deepseek"]
MAX_ROLLOUTS, SUBSAMPLE_SEED = 50, 42
BFCL_CATS = {"base": "multi_turn_base", "longctx": "multi_turn_long_context",
             "missfunc": "multi_turn_miss_func", "missparam": "multi_turn_miss_param"}
SOURCE_NOTE = {
    ("tau2", "gpt5mini"): "exp/models/gpt5mini_200/data (200 tasks x 50, run_gpt5mini_200.sh)",
    ("tau2", "deepseek"): "exp/models/dsv32_200/data (200 tasks x 50, run_dsv32_200.sh)",
    ("tau2", "qwen"): "exp/tau2/baseline/consolidated_baselines/data (consolidated from 8 Qwen runs; manifest.json has per-rollout provenance)",
    ("bfclv3", "gpt5mini"): "exp/bfcl/runs/gpt5mini_repro_200x50; bundled-step embeddings (table1_data_gpt5mini)",
    ("bfclv3", "qwen"): "exp/bfcl/runs/400_across_cats (200 preset tasks of 400); per-call embeddings (table1_data_qwen_v2)",
    ("bfclv3", "deepseek"): "exp/bfcl/runs/dsv32_194_prompt; per-call embeddings (table1_data_dsv32_v2)",
    ("agentclinic", "gpt5mini"): "benchmark_refs/AgentClinic/outputs/medqa-ext-gpt5mini-50rollouts; v2 (extracted-action) embeddings",
    ("agentclinic", "qwen"): "benchmark_refs/AgentClinic/outputs/medqa-ext-qwen-50rollouts; v2 embeddings",
    ("agentclinic", "deepseek"): "benchmark_refs/AgentClinic/outputs/medqa-ext-deepseek-10rollouts (expanded to 50); v2 embeddings",
}


def subsample(rids):
    rids = sorted(rids)
    if len(rids) <= MAX_ROLLOUTS:
        return rids, []
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    keep = set(rng.choice(rids, size=MAX_ROLLOUTS, replace=False).tolist())
    return sorted(keep), [r for r in rids if r not in keep]


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(os.path.realpath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_embeddings(cell_dir, stem, out_task, keep_rids, filter_keys):
    act = cell_dir / "embeddings" / "action" / f"{stem}.npz"
    res = cell_dir / "embeddings" / "response" / f"{stem}.npz"
    if not filter_keys:
        link_or_copy(act, out_task / "action.npz")
        link_or_copy(res, out_task / "observation.npz")
        return
    out_task.mkdir(parents=True, exist_ok=True)
    keep = {f"r{r}_" for r in keep_rids}
    for src, name in ((act, "action.npz"), (res, "observation.npz")):
        if (out_task / name).exists():
            continue
        z = np.load(src)
        arrays = {k: z[k] for k in z.files if any(k.startswith(p) for p in keep)}
        np.savez(out_task / name, **arrays)
        z.close()


def npz_step_counts(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        m = re.match(r"r(\d+)_s(\d+)$", k)
        if m:
            out[int(m.group(1))] = max(out.get(int(m.group(1)), -1), int(m.group(2)))
    z.close()
    return {r: s + 1 for r, s in out.items()}


def envelope(benchmark, policy, task, rid, reward, steps, raw):
    return {"benchmark": benchmark, "policy": POLICY_DIR[policy], "task": task, "rid": rid,
            "reward": reward, "n_steps": len(steps),
            "steps": [{"action": a, "observation": o} for a, o in steps], "raw": raw}


def emit_task(out, benchmark, policy, task, rollouts, cell_dir, stem, meta):
    """rollouts: {rid: (reward, steps, raw)}. Writes rollout files, extra jsonl,
    task.json and the per-task embeddings; checks step counts against the npz."""
    keep, extra = subsample(rollouts)
    rdir = out / "rollouts" / benchmark / POLICY_DIR[policy] / task
    edir = out / "embeddings" / benchmark / POLICY_DIR[policy] / task
    rdir.mkdir(parents=True, exist_ok=True)
    for r in keep:
        reward, steps, raw = rollouts[r]
        p = rdir / f"rollout_{r:03d}.json"
        if not p.exists():
            json.dump(envelope(benchmark, policy, task, r, reward, steps, raw), open(p, "w"))
    if extra:
        p = rdir / "extra_rollouts.jsonl"
        if not p.exists():
            with open(p, "w") as fh:
                for r in extra:
                    reward, steps, raw = rollouts[r]
                    fh.write(json.dumps(envelope(benchmark, policy, task, r, reward, steps, raw)) + "\n")
    # step-count check against the source npz BEFORE writing the embeddings
    src_counts = npz_step_counts(cell_dir / "embeddings" / "action" / f"{stem}.npz")
    bad = [r for r in keep if src_counts.get(r) != len(rollouts[r][1])]
    note = None
    if bad:
        # Known case: tau2/qwen retail/31 and retail/42, whose consolidated embeddings were
        # copied from an older run and no longer match the rollout files. Both tasks are
        # degenerate under the 50-rollout subsample and enter no result; ship rollouts only.
        note = (f"embeddings omitted: the source embeddings were stale for {len(bad)}/{len(keep)} "
                f"rollouts; this task is not used by any result")
        shutil.rmtree(edir, ignore_errors=True)
        print(f"  [warn] {benchmark}/{policy}/{task}: {note}", flush=True)
    else:
        write_embeddings(cell_dir, stem, edir, keep, filter_keys=bool(extra))
        json.dump({str(r): rollouts[r][0] for r in keep}, open(edir / "rewards.json", "w"), indent=0)
        counts = npz_step_counts(edir / "action.npz")
        if set(counts) != set(keep):
            raise SystemExit(f"rid mismatch after write {benchmark}/{policy}/{task}")
    sr = float(np.mean([rollouts[r][0] >= 0.5 for r in keep]))
    info = {"task": task, "benchmark": benchmark, "policy": POLICY_DIR[policy], **meta,
            "n_rollouts": len(keep), "n_extra_rollouts": len(extra), "success_rate": round(sr, 4),
            "source": SOURCE_NOTE[(benchmark, policy)]}
    if note:
        info["embeddings"] = note
    json.dump(info, open(rdir / "task.json", "w"), indent=1)
    return len(keep), len(extra)


# ------------------------------------------------------------------ tau2
def build_tau2(out, policy):
    from src.prm.embeddings import extract_response_text, extract_sample_text
    cell = TABLE1 / f"tau2_{policy}"
    tel = json.load(open(ROLLOUTS / "meta" / "tau2_telecom_id_mapping.json"))
    idx_to_orig = {v.split("/", 1)[1]: k.split("/", 1)[1] for k, v in tel.items()}
    n_tasks = n_r = n_x = 0
    for dom in ("airline", "retail", "telecom"):
        for tdir in sorted((cell / dom).iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            if not tdir.is_dir():
                continue
            rollouts = {}
            for rf in sorted(tdir.glob("rollout_*.json")):
                rid = int(rf.stem.split("_")[1])
                raw = json.load(open(rf))
                all_steps = raw.get("steps", [])
                steps = []
                for i, st in enumerate(all_steps):
                    acts = [extract_sample_text(s, "action") for s in st.get("samples", [])]
                    obs = extract_response_text(all_steps, i, messages=raw.get("messages"))
                    steps.append((acts[0] if len(acts) == 1 else acts, obs))
                rollouts[rid] = (float(raw.get("reward", 0.0)), steps, raw)
            task = f"{dom}_{tdir.name}"
            meta = {"domain": dom, "task_id": f"{dom}/{idx_to_orig.get(tdir.name, tdir.name)}" if dom == "telecom" else f"{dom}/{tdir.name}"}
            k, x = emit_task(out, "tau2", policy, task, rollouts, cell, task, meta)
            n_tasks += 1; n_r += k; n_x += x
    print(f"tau2/{policy}: {n_tasks} tasks, {n_r} rollouts, {n_x} extra", flush=True)


# ------------------------------------------------------------------ bfclv3
def _bfcl_raw(rdir):
    """Collapse one gorilla rollout directory into a JSON-able record."""
    rec = {"result": [], "score": []}
    for f in sorted(rdir.glob("result/*/multi_turn/*.json")):
        rec["result"] += [json.loads(l) for l in open(f) if l.strip()]
    for f in sorted(rdir.glob("score/*/multi_turn/*.json")):
        rec["score"] += [json.loads(l) for l in open(f) if l.strip()]
    return rec


def build_bfcl(out, policy):
    from generation.bfcl.embed_bfcl_percall import parse_task_rollouts_v2
    from generation.bfcl.embed_bfcl_bundled import parse_task_rollouts_v3
    parse = parse_task_rollouts_v3 if policy == "gpt5mini" else parse_task_rollouts_v2
    cell = TABLE1 / f"bfclv3_{policy}"
    raw_root = ROLLOUTS / "raw" / "bfclv3" / {"gpt5mini": "gpt5mini", "qwen": "qwen", "deepseek": "deepseek"}[policy]
    n_tasks = n_r = 0
    for short, cat in BFCL_CATS.items():
        for stub in sorted((cell / short).glob("*_tdp_00.json"), key=lambda p: int(p.name.split("_")[0])):
            tid = stub.name.split("_tdp_")[0]
            task_id = f"{cat}_{tid}"
            tdir = raw_root / cat / task_id
            data = parse(tdir, task_id)
            per = {}
            for (rid, _), a, o in zip(data["indices"], data["action_texts"], data["response_texts"]):
                per.setdefault(rid, []).append((a, o))
            rollouts = {}
            for rid, steps in per.items():
                reward = float(data["rewards"].get(rid, 0.0))
                rollouts[rid] = (reward, steps, _bfcl_raw(tdir / f"rollout_{rid:02d}"))
            task = f"{short}_{tid}"
            k, _ = emit_task(out, "bfclv3", policy, task, rollouts, cell, task,
                             {"category": cat, "task_id": task_id,
                              "step_granularity": "bundled" if policy == "gpt5mini" else "per_call"})
            n_tasks += 1; n_r += k
    print(f"bfclv3/{policy}: {n_tasks} tasks, {n_r} rollouts", flush=True)


# ------------------------------------------------------------------ agentclinic
def build_agentclinic(out, policy):
    from generation.agentclinic.embed_agentclinic import _extract_action_response
    cell = TABLE1 / f"agentclinic_{policy}"
    raw_root = ROLLOUTS / "raw" / "agentclinic" / {"gpt5mini": "gpt5mini", "qwen": "qwen", "deepseek": "deepseek"}[policy]
    n_tasks = n_r = 0
    sids = sorted({int(p.name.split("_tdp_")[0]) for p in (cell / "medqa").glob("*_tdp_*.json")})
    for sid in sids:
        rollouts = {}
        for rf in sorted(raw_root.glob(f"scenario_{sid:04d}_rollout_*.json")):
            rid = int(rf.stem.split("_rollout_")[1])
            raw = json.load(open(rf))
            acts, obs = _extract_action_response(raw.get("history", []), action_mode="action")
            rollouts[rid] = (1.0 if raw.get("correct") else 0.0, list(zip(acts, obs)), raw)
        task = f"scenario_{sid:04d}"
        k, _ = emit_task(out, "agentclinic", policy, task, rollouts, cell, f"medqa_{sid}",
                         {"dataset": "MedQA_Ext", "scenario_id": sid,
                          "correct_diagnosis": next(iter(rollouts.values()))[2].get("correct_diagnosis")})
        n_tasks += 1; n_r += k
    print(f"agentclinic/{policy}: {n_tasks} scenarios, {n_r} rollouts", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", nargs="+", choices=["tau2", "bfclv3", "agentclinic"], required=True)
    ap.add_argument("--policy", nargs="*", default=MODELS, choices=MODELS)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    for b in args.benchmark:
        for p in args.policy:
            {"tau2": build_tau2, "bfclv3": build_bfcl, "agentclinic": build_agentclinic}[b](args.out, p)


if __name__ == "__main__":
    main()
