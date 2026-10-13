#!/usr/bin/env python3
"""Resample the final diagnosis given a fixed diagnostic prefix.

Question: is a rollout's outcome determined by its trajectory, or decided
(stochastically) only at the final synthesis step? For each selected rollout we
reconstruct the doctor's context up to — but not including — its final DIAGNOSIS
turn, then resample that final diagnosis N times from the same model and grade
each. If the resampled outcome is consistent, the prefix determines the outcome
(a perfect discriminator could predict it). If it flips, the trajectory has
little bearing on correctness — the outcome is decided at the last step.

Design: <n_tasks> mixed tasks x <n_rollouts> rollouts x <n_samples> resamples.

Usage (run from benchmark_refs/AgentClinic with keys exported):
    python resample_diagnosis.py --doctor_llm vertex-deepseek-v3.2 \
        --rollouts_dir outputs/medqa-ext-deepseek-10rollouts --out resample_ds.json
    python resample_diagnosis.py --doctor_llm gpt-5-mini \
        --rollouts_dir outputs/medqa-ext-gpt5mini-50rollouts --out resample_gpt.json
"""
import argparse
import glob
import json
import os
import re
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from agentclinic import (
    DoctorAgent, ScenarioLoaderMedQAExtended, compare_results, query_model,
)

FINAL_Q = "This is the final question. Please provide a diagnosis.\n"


def reconstruct_prefix(history, total_inferences=20):
    """Return (agent_hist, question, infs) for the doctor's final diagnosis turn,
    or None if the rollout didn't end in a diagnosis."""
    doctor_idxs = [i for i, t in enumerate(history) if t.get("role") == "doctor"]
    if not doctor_idxs:
        return None
    final_i = doctor_idxs[-1]
    if "diagnosis" not in str(history[final_i].get("content", "")).lower():
        return None

    agent_hist, pi, infs = "", "", 0
    for i, turn in enumerate(history):
        if turn.get("role") == "doctor":
            if i == final_i:
                break
            agent_hist += pi + "\n\n" + str(turn.get("content", "")) + "\n\n"
            infs += 1
        else:
            pi = str(turn.get("content", ""))
    question = pi
    # Force-all: append FINAL_Q on EVERY rollout (matching generation's placement,
    # right before the doctor answers) so all models are forced identically.
    question = question + FINAL_Q
    return agent_hist, question, infs


def resample_rollout(scenario, backend, agent_hist, question, infs,
                     correct_diagnosis, moderator_llm, n_samples):
    doc = DoctorAgent(scenario, backend_str=backend, max_infs=20)
    doc.infs = infs
    sys_prompt = doc.system_prompt()
    prompt = ("\nHere is a history of your dialogue: " + agent_hist +
              "\n Here was the patient response: " + question +
              "Now please continue your dialogue\nDoctor: ")
    results = []
    for _ in range(n_samples):
        ans = query_model(backend, prompt, sys_prompt, scene=scenario)
        verdict = compare_results(ans, correct_diagnosis, moderator_llm, None)
        results.append({"diagnosis": ans[:300], "correct": verdict.strip().startswith("yes")})
    return results


def pick_tasks(rollouts_dir, sr_lo, sr_hi, n_tasks, seed):
    by_scen = defaultdict(list)
    for fp in glob.glob(f"{rollouts_dir}/scenario_*_rollout_*.json"):
        sid = int(re.search(r"scenario_(\d+)_", fp).group(1))
        by_scen[sid].append(fp)
    eligible = []
    for sid, files in by_scen.items():
        outs = []
        for fp in files:
            d = json.load(open(fp))
            outs.append(1 if d.get("correct") else 0)
        sr = np.mean(outs)
        if sr_lo <= sr <= sr_hi:
            eligible.append(sid)
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return sorted(eligible[:n_tasks]), by_scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doctor_llm", required=True)
    ap.add_argument("--rollouts_dir", required=True)
    ap.add_argument("--moderator_llm", default="gpt-4o-mini")
    ap.add_argument("--n_tasks", type=int, default=20)
    ap.add_argument("--n_rollouts", type=int, default=10)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--sr_lo", type=float, default=0.3)
    ap.add_argument("--sr_hi", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loader = ScenarioLoaderMedQAExtended()
    task_ids, by_scen = pick_tasks(args.rollouts_dir, args.sr_lo, args.sr_hi,
                                   args.n_tasks, args.seed)
    print(f"Selected {len(task_ids)} mixed tasks: {task_ids}")

    # build job list: (sid, rollout_file)
    jobs = []
    rng = random.Random(args.seed)
    for sid in task_ids:
        files = sorted(by_scen[sid])
        # keep only rollouts that ended in a diagnosis
        usable = [fp for fp in files
                  if reconstruct_prefix(json.load(open(fp)).get("history", [])) is not None]
        rng.shuffle(usable)
        for fp in usable[:args.n_rollouts]:
            jobs.append((sid, fp))
    print(f"{len(jobs)} rollouts to resample x {args.n_samples} samples "
          f"= {len(jobs) * args.n_samples} diagnosis generations")

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))

    def run_job(sid, fp):
        key = os.path.basename(fp)
        d = json.load(open(fp))
        rec = reconstruct_prefix(d.get("history", []))
        if rec is None:
            return None
        agent_hist, question, infs = rec
        scenario = loader.get_scenario(sid)
        correct = str(d.get("correct_diagnosis") or scenario.diagnosis_information())
        samples = resample_rollout(scenario, args.doctor_llm, agent_hist, question,
                                   infs, correct, args.moderator_llm, args.n_samples)
        n_corr = sum(s["correct"] for s in samples)
        return key, {
            "scenario_id": sid,
            "orig_correct": bool(d.get("correct")),
            "n_samples": len(samples),
            "n_resampled_correct": n_corr,
            "resample_correct_rate": n_corr / len(samples),
            "consistent": (n_corr == 0 or n_corr == len(samples)),
            "samples": samples,
        }

    todo = [(sid, fp) for sid, fp in jobs if os.path.basename(fp) not in results]
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_job, sid, fp): fp for sid, fp in todo}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results[r[0]] = r[1]
                done += 1
                if done % 5 == 0:
                    json.dump(results, open(args.out, "w"), indent=2)
                    print(f"\r  {done}/{len(todo)} rollouts resampled", end="", flush=True)
    json.dump(results, open(args.out, "w"), indent=2)

    # summary
    rr = list(results.values())
    flip = [r for r in rr if not r["consistent"]]
    print(f"\n\n=== {args.doctor_llm} ===")
    print(f"rollouts: {len(rr)}")
    print(f"prefix-consistent (all resamples same outcome): "
          f"{len(rr) - len(flip)}/{len(rr)} = {(len(rr)-len(flip))/len(rr):.0%}")
    print(f"prefix-INDETERMINATE (resamples flip): {len(flip)}/{len(rr)} = "
          f"{len(flip)/len(rr):.0%}")
    rates = np.array([r["resample_correct_rate"] for r in rr])
    print(f"resample-correct-rate distribution: "
          f"mean={rates.mean():.2f} | at 0: {(rates==0).sum()} | "
          f"mid(0.2-0.8): {((rates>=0.2)&(rates<=0.8)).sum()} | at 1: {(rates==1).sum()}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
