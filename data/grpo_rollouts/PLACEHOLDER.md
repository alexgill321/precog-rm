# Raw GRPO rollout logs (HuggingFace data release; optional)

    <run_name>/rollouts/batch_<n>.jsonl    full rollout records per batch
    <run_name>/metrics.jsonl               (already included in this release)

Run names: the 21 official runs listed in scripts/fig_grpo_curves.py
(VARIANTS). ~56 GB. Only needed to rebuild the per-rollout token cache
(notebooks/data/tau2_es/rollout_cache/) from scratch or to inspect
trajectories; every Table-3 number rebuilds from the included caches.
