---
license: mit
tags: [llm-agents, process-reward-model, prototypical-networks, meta-learning]
pretty_name: Precog-RM checkpoints
---

# Precog-RM checkpoints

Trained encoders from *"Divergent Paths: Learning an Outcome-Predictive Representation
of LLM Agent Trajectories"* (Gill, Gogia, Guzmán), [arXiv:2610.19614](http://arxiv.org/abs/2610.19614). Code and loaders:
https://github.com/alexgill321/precog-rm (`src/prm/train_precog.py:PrecogProtoNet.load`).

```
precog/precog_<pool>_s<seed>.pt      75 checkpoints of the main sweep (25 training pools x seeds 0-2):
                                     <benchmark>_<g|q|d combination> (g=GPT-5-mini, q=Qwen3-80B, d=DeepSeek-V3.2),
                                     two-benchmark and generalist pools as <b1>-<b2>[-<b3>]_all.
                                     10,000 episodes each; Tables 1-2, Figure 3, Appendix J.
ablations/src_<name>_s0.pt           single-seed design study on tau2/qwen (Appendices C, F; 4,000 episodes)
ablations/mix_<name>_s0.pt           the same ablations retrained on the all-cell pool
ablations/pool_<pool>_full_s0.pt     epoch-resolved pool sweep (notebook study)
prefix/prefix_<pool>_<variant>_s0.pt prefix-trained encoders (Section 4.5, Figure 4)
prefix/grpo_controller_model_prefix.pt  the controller deployed inside GRPO (Section 5); trained on
                                     tau2/qwen with scripts/run_prefix_verify.py (predates the final sweep protocol)
```

Architecture: linear projection of 3072-d step embeddings to d'=128, 2-layer transformer
(4 heads, FFN 512, dropout 0.3), masked mean pooling, unsquared-Euclidean prototypes.
Produced by `scripts/precog_sweep.py`, `scripts/design_study.py --train`,
`scripts/prefix_gating_eval.py --train`.
