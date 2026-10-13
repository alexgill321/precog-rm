#!/usr/bin/env python3
"""Expand the 9 mixed-SR terra MedQA scenarios from 10 -> 50 rollouts each.
Targets specific scenario IDs (agentclinic.py only supports first-N), rollout
IDs 10..49. Resumable: _run_single skips existing files. terra = gpt-5.6-terra
via api.openai.com (no Vertex contention)."""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import agentclinic as ac

MIXED = [1, 2, 3, 5, 7, 14, 15, 16, 18]
OUT = "outputs/medqa-ext-gpt56terra-10rollouts"
CONC = 20
loader = ac.ScenarioLoaderMedQAExtended()
jobs = [(sid, rid) for sid in MIXED for rid in range(10, 50)]
print(f"{len(jobs)} rollouts to generate (9 scenarios x 40) with gpt-5.6-terra")

def run(job):
    sid, rid = job
    sc = loader.get_scenario(sid)
    r = ac._run_single(sid, rid, sc, "MedQA_Ext", "llm", "None", "None",
                       "gpt-5.6-terra", "gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini",
                       total_inferences=20, img_request=False, output_dir=OUT)
    return sid, rid, (r.get("correct") if r else "skip"), (r.get("cost_usd") if r else 0)

done = 0; cost = 0.0
with ThreadPoolExecutor(CONC) as ex:
    for fut in as_completed([ex.submit(run, j) for j in jobs]):
        sid, rid, correct, c = fut.result()
        cost += c or 0; done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(jobs)}  cumulative ${cost:.2f}", flush=True)
print(f"done: {done} rollouts, total cost ${cost:.2f}")
