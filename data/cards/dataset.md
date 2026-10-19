---
license: mit
task_categories: [other]
tags: [llm-agents, trajectories, embeddings, process-reward-model, grpo, tau2-bench, bfcl, agentclinic]
pretty_name: Precog-RM data release
size_categories: [10B<n<100B]
---

# Precog-RM data release

Everything behind *"Divergent Paths: Learning an Outcome-Predictive Representation of
LLM Agent Trajectories"* (Gill, Gogia, Guzmán), [arXiv:2610.19614](http://arxiv.org/abs/2610.19614). Code and replication guide:
https://github.com/alexgill321/precog-rm. Checkpoints: `agill32/precog-rm-checkpoints`.

Four parts, browsable top-down. Benchmarks: `tau2` (Tau2-Bench), `bfclv3`
(Berkeley Function Calling Leaderboard v3 multi-turn), `agentclinic` (MedQA-Extended).
Policies: `gpt5mini` (gpt-5-mini-2025-08-07, reasoning_effort=minimal), `qwen3-80b`
(Qwen3-Next-80B-Thinking), `deepseek-v3.2`.

```
rollouts/<benchmark>/<policy>/<task>/
    rollout_000.json ... rollout_049.json   the 50 rollouts the paper used for this task
    extra_rollouts.jsonl                    surplus rollouts (tau2/qwen3-80b only; not used by any result)
    task.json                               task id, category, success rate, source run
embeddings/<benchmark>/<policy>/<task>/
    action.npz                              keys r<rid>_s<step>, float32 (3072,), text-embedding-3-large
    observation.npz                         same keys; the environment side of each step
    rewards.json                            {rid: reward}
embeddings/reasoning/                       per-step reasoning embeddings, tau2/qwen3-80b test tasks (App. B.1)
embeddings/grpo_replay/                     baseline-GRPO rollouts embedded per batch (simulated early stopping)
grpo/runs/<run>/                            21 official GRPO runs: metrics.jsonl, rollouts/batch_*.jsonl, support_sets/
grpo/replay/, grpo/eval/                    replay results, base-model eval
caches/mmap/                                optional float16 memory-mapped caches (derived; skips the HDBSCAN rebuild)
ablations/agentclinic/                      no-tools ablation (tarballs), final-diagnosis resamples, GPT-5.6-Terra reference
meta/                                       task lists, telecom id mapping, tool-reliance scenario lists
```

Each rollout JSON has one envelope: `benchmark, policy, task, rid, reward, n_steps`,
`steps` (the list of `{action, observation}` texts in exactly the order they were
embedded, so step *t* is key `r<rid>_s<t>` in the npz files), and `raw` (the
untouched harness record: tau2 messages and reward info, gorilla result and score
outputs, AgentClinic history/tokens/cost). Tasks: 200 per policy for tau2 and
bfclv3, 214 scenarios for agentclinic; 50 rollouts each. Provenance per task is in
`task.json["source"]`.

Download by part with the repo's `scripts/download.py`, which also lays the
embeddings out the way the code reads them. Benchmarks: tau2-bench (MIT),
gorilla/BFCL (Apache-2.0), AgentClinic (MIT; MedQA-Extended scenarios). Simulated
users and patients are gpt-4o-mini.
