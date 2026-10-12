#!/usr/bin/env python3
"""Appendix I (tab:ac-resample): final-diagnosis resampling on AgentClinic.

Reads $PRECOG_DATA/rollouts/resample/<policy>.json, written by
benchmark_refs/AgentClinic/resample_diagnosis.py: for each selected rollout the
doctor's context up to (excluding) its final DIAGNOSIS turn is replayed and the
diagnosis resampled 10 times from the same policy; each resample is graded by
the environment's grader.

With p_hat = fraction of a rollout's resamples graded correct:
  Consistent    = fraction of rollouts with p_hat in {0, 1}
  Ambiguous     = fraction with 0.2 < p_hat < 0.8
  AUROC ceiling = expected per-task AUROC of an oracle that scores each
                  rollout by its own p_hat, with outcomes drawn from those same
                  p_hat (closed form: pairwise P(score_i > score_j) weighted by
                  p_i (1 - p_j), ties count 1/2), averaged over scenarios.

    python scripts/resample_summary.py
"""
from collections import defaultdict
import json

import numpy as np

from _paths import ROLLOUTS, require

POLICIES = [("deepseek", "DeepSeek-V3.2"), ("qwen", "Qwen3-Next-80B-Thinking"),
            ("gpt5mini", "GPT-5-mini"), ("gpt56terra", "GPT-5.6-Terra")]


def auroc_ceiling(ps):
    ps = np.asarray(ps, dtype=float)
    num = den = 0.0
    for i, pi in enumerate(ps):
        for j, pj in enumerate(ps):
            if i == j:
                continue
            w = pi * (1 - pj)
            den += w
            num += w * (1.0 if pi > pj else 0.5 if pi == pj else 0.0)
    return num / den if den > 0 else np.nan


def summarize(path):
    d = json.load(open(path))
    rates = np.array([v["resample_correct_rate"] for v in d.values()])
    by_task = defaultdict(list)
    for v in d.values():
        by_task[v["scenario_id"]].append(v["resample_correct_rate"])
    ceil = np.nanmean([auroc_ceiling(ps) for ps in by_task.values()])
    return dict(n_rollouts=len(rates), n_tasks=len(by_task),
                consistent=float(np.mean((rates == 0) | (rates == 1))),
                ambiguous=float(np.mean((rates > 0.2) & (rates < 0.8))),
                auroc_ceiling=float(ceil))


def main():
    d = require(ROLLOUTS / "resample", "resample JSONs", "rollouts")
    print(f"{'policy':26} {'rollouts':>8} {'tasks':>5} {'Cons.':>6} {'Ambig.':>6} {'AUROC ceil.':>11}")
    for key, name in POLICIES:
        p = d / f"{key}.json"
        if not p.exists():
            print(f"{name:26} missing: {p}")
            continue
        s = summarize(p)
        print(f"{name:26} {s['n_rollouts']:8d} {s['n_tasks']:5d} {s['consistent']:6.2f} "
              f"{s['ambiguous']:6.2f} {s['auroc_ceiling']:11.3f}")


if __name__ == "__main__":
    main()
