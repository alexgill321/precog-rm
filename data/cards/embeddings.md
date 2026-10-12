---
license: mit
task_categories: [other]
tags: [llm-agents, trajectories, embeddings, process-reward-model, tau2-bench, bfcl, agentclinic]
pretty_name: Precog-RM embedding corpora
size_categories: [10B<n<100B]
---

# Precog-RM embedding corpora

Step-level embeddings of agent trajectories used in *"Divergent Paths: Learning an
Outcome-Predictive Representation of LLM Agent Trajectories"* (Gill, Gogia, Guzmán).
Code: https://github.com/alexgill321/precog-rm

Nine (benchmark, policy) cells: {Tau2-Bench, BFCLv3, AgentClinic MedQA-Extended} x
{DeepSeek-V3.2, Qwen3-Next-80B-Thinking, GPT-5-mini}. For every rollout step the
action text and the observation text were embedded independently with OpenAI
`text-embedding-3-large` (d=3072). Up to 50 rollouts per task (the tau2/qwen cell
holds more; loaders cap at 50). Extraction rules per benchmark are in Appendix D of
the paper.

```
table1_data/<benchmark>_<model>/
    embeddings/action/<task>.npz      keys r<rid>_s<step>, float32 (3072,)
    embeddings/response/<task>.npz    observation embeddings, same keys
    <category>/<task>_tdp_<rid>.json  reward stubs (bfclv3, agentclinic)
    <domain>/<idx>/rollout_<rid>.json full rollouts with messages (tau2 cells)
    manifest.json                     tau2_qwen: per-rollout rewards and provenance
reasoning_emb/tau2_qwen_steps/        per-step reasoning embeddings, tau2/qwen test tasks
grpo_replay/official_retail_s{1,2,3}/ baseline-GRPO rollouts (Qwen3-8B) embedded per batch,
                                      used by the simulated early-stopping study
caches/mmap/                          float16 memory-mapped caches (optional fast path;
                                      rebuilt by scripts/build_slim_caches.py + --convert)
```

Pipeline entry: `python scripts/build_slim_caches.py && python scripts/pool_traj_worker.py --convert`
(or download `caches/` to skip it), then every experiment in the repo.

Benchmarks: tau2-bench (MIT), gorilla/BFCL (Apache-2.0), AgentClinic (MedQA-derived
scenarios). Simulated users/patients are gpt-4o-mini. See the repo's REPLICATION.md
for which paper result reads which part of this release.
