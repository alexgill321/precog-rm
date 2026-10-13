#!/usr/bin/env python3
"""Appendix D/I: AgentClinic tool-reliance labels from the no-tools ablation.

For each policy, every mixed-outcome MedQA-Extended scenario was re-run with a
doctor prompt stripped of the ability to order tests (25 rollouts; raw records
under $PRECOG_DATA/ablations/agentclinic/notools/<policy>/). A scenario is
tool-reliant if its success rate drops by more than 0.1 without tools: one-sided
Farrington-Manning score test of H1: SR_with - SR_without > 0.1 against the
with-tools 50-rollout baseline, Benjamini-Hochberg FDR 5% within each policy.
Tool-reliant and non-tool lists partition the mixed-outcome scenarios
(success rate in [0.16, 0.84]).

Prints the counts quoted in Appendix I and writes the label lists to
benchmark_refs/AgentClinic/ (the files consumed by the evaluation protocol).
`--check` compares against the shipped lists instead of overwriting them.

    python scripts/notools_analysis.py --check
"""
import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from _paths import ABLATIONS, TABLE1, AC_META, MODEL_NAME, POLICY_DIR, require

SR_LO, SR_HI, MARGIN, FDR = 0.16, 0.84, 0.10, 0.05
LABEL_FILES = {"gpt5mini": ("toolreliant_26.txt", "nontool_mixed.txt"),
               "deepseek": ("toolreliant_deepseek_fdr_mixed.txt", "nontool_deepseek_mixed.txt"),
               "qwen": ("toolreliant_qwen_fdr_mixed.txt", "nontool_qwen_mixed.txt")}


def fm_margin_p(x1, n1, x2, n2, delta):
    """One-sided Farrington-Manning: H0 p1-p2=delta vs H1 p1-p2>delta (group 1 = with tools)."""
    p1h, p2h = x1 / n1, x2 / n2
    if (p1h - p2h) <= delta:
        return 1.0
    def nll(p2):
        p1 = p2 + delta; e = 1e-10
        p1 = min(max(p1, e), 1 - e); p2c = min(max(p2, e), 1 - e)
        return -(x1 * np.log(p1) + (n1 - x1) * np.log(1 - p1) + x2 * np.log(p2c) + (n2 - x2) * np.log(1 - p2c))
    lo, hi = max(0., -delta) + 1e-9, min(1., 1. - delta) - 1e-9
    p2t = minimize_scalar(nll, bounds=(lo, hi), method="bounded").x
    p1t = p2t + delta
    se = np.sqrt(p1t * (1 - p1t) / n1 + p2t * (1 - p2t) / n2)
    return float(1 - norm.cdf((p1h - p2h - delta) / se))


def with_tools_outcomes(policy):
    """{scenario_id: [0/1 per rollout]} from the table1_data reward stubs of the cell."""
    d = TABLE1 / f"agentclinic_{policy}" / "medqa"
    out = defaultdict(list)
    for f in sorted(glob.glob(str(d / "*_tdp_*.json"))):
        sid = int(f.split("/")[-1].split("_tdp_")[0])
        r = json.load(open(f))
        rew = r.get("reward", r.get("correct"))
        out[sid].append(1 if float(rew) >= 0.5 else 0)
    return out


def notools_outcomes(policy):
    out = defaultdict(list)
    for f in glob.glob(str(ABLATIONS / "agentclinic" / "notools" / POLICY_DIR[policy] / "scenario_*_rollout_*.json")):
        x = json.load(open(f)); out[int(x["scenario_id"])].append(int(bool(x["correct"])))
    return out


def label_policy(policy):
    base, abl = with_tools_outcomes(policy), notools_outcomes(policy)
    rows = []
    for sid, a in abl.items():
        if sid not in base:
            continue
        b = base[sid]
        rows.append((sid, sum(b) / len(b), sum(a) / len(a), fm_margin_p(sum(b), len(b), sum(a), len(a), MARGIN)))
    ps = np.array([r[3] for r in rows]); m = len(ps); order = np.argsort(ps); bh = 0.0
    for rank, idx in enumerate(order, 1):
        if ps[idx] <= FDR * rank / m:
            bh = ps[idx]
    mixed = {r[0] for r in rows if SR_LO <= r[1] <= SR_HI}
    tool = sorted({r[0] for r in rows if r[3] <= bh} & mixed)
    nontool = sorted(mixed - set(tool))
    return tool, nontool, sorted(mixed), len(rows), float(np.mean([r[2] - r[1] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with shipped lists; do not overwrite")
    args = ap.parse_args()
    require(ABLATIONS / "agentclinic" / "notools", "no-tools rollouts", "ablations")
    print(f"{'policy':11} {'scenarios':>9} {'mixed':>5} {'tool-reliant':>12} {'no confirmed drop':>18} {'macro dSR':>9}")
    for policy, (tf, nf) in LABEL_FILES.items():
        tool, nontool, mixed, n, dsr = label_policy(policy)
        print(f"{MODEL_NAME[policy]:11} {n:9d} {len(mixed):5d} {len(tool):12d} "
              f"{len(nontool):6d} ({len(nontool) / len(mixed):.0%}) {dsr:+9.3f}")
        if args.check:
            old = {int(x) for x in (AC_META / tf).read_text().split(",") if x.strip()}
            print(f"    shipped {tf}: {len(old)} ids; {'identical' if old == set(tool) else 'DIFFERENT: ' + str(sorted(old ^ set(tool)))}")
        else:
            (AC_META / tf).write_text(",".join(map(str, tool)))
            (AC_META / nf).write_text(",".join(map(str, nontool)))
    if not args.check:
        print(f"label files written under {AC_META}")


if __name__ == "__main__":
    main()
