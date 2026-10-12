# Precog-RM checkpoints (HuggingFace data release)

    ckpts/precog_<pool>_s<seed>.pt

Pools: <benchmark>_<models> with models in {g,q,d} joined by '-'
(e.g. tau2_g-q-d), two-benchmark and generalist pools as
<b1>-<b2>[-<b3>]_all. Seeds 0-2. Table 1 uses the per-benchmark
specialists (<benchmark>_g-q-d) and the generalist
(agentclinic-bfclv3-tau2_all); scripts/merged_cv_table.py expects them at
exp/results/protonet_ablations/ckpts/. Retraining from scratch:
scripts/precog_sweep.py.
