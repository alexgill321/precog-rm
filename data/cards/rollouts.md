---
license: mit
tags: [llm-agents, trajectories, tau2-bench, bfcl, agentclinic]
pretty_name: Precog-RM raw rollouts
size_categories: [1B<n<10B]
---

# Precog-RM raw rollouts

Raw agent trajectories behind *"Divergent Paths: Learning an Outcome-Predictive
Representation of LLM Agent Trajectories"* (Gill, Gogia, Guzman), for re-embedding,
inspection, and the AgentClinic analyses of Appendix I.
Code: https://github.com/alexgill321/precog-rm

```
raw/bfclv3/<model>/<category>/<task>/rollout_<rid>/{result,score}/   BFCLv3 multi-turn runs,
                                     200 tasks x 50 rollouts per policy (gorilla checker outputs)
raw/agentclinic/<model>/scenario_<sid>_rollout_<rid>.json            MedQA-Extended, 214 scenarios x 50
raw/agentclinic/gpt56terra/          GPT-5.6-Terra, 9 mixed scenarios x 50 (resample reference policy)
raw/agentclinic_notools/<model>/     no-tools ablation (REQUEST TEST removed), 25 rollouts per scenario
resample/<model>.json                final-diagnosis resampling: 20 scenarios x 10 rollouts x 10 resamples
meta/                                tool-reliant / non-tool scenario lists, tau2 task ids, BFCL task list
```

The tau2 rollouts (full message logs for all three policies) ship inside the
embedding corpora (`precog-rm-embeddings`, `table1_data/tau2_*`).

Policies: DeepSeek-V3.2 and Qwen3-Next-80B-Thinking via Vertex MaaS, GPT-5-mini
(reasoning effort minimal) via the OpenAI API; simulated users / patients /
measurement agents are gpt-4o-mini. Benchmarks: tau2-bench (MIT), gorilla/BFCL
(Apache-2.0), AgentClinic (MedQA-Extended scenarios; see the AgentClinic repository
for the scenario license).

Scripts: `notools_analysis.py` (tool-reliance labels, Appendix I counts),
`resample_summary.py` (Table: final-diagnosis resampling), `fig_trajectory_flow.py`.
