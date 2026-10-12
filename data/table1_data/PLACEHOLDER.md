# Embedding corpora (HuggingFace data release)

Nine directories, one per (benchmark, model) cell:

    table1_data/<benchmark>_<model>/          benchmark in {tau2, bfclv3, agentclinic}
                                              model in {gpt5mini, qwen, deepseek}
        <category>/<task>_tdp_<rid>.json      per-rollout reward stubs
        embeddings/action/<task>.npz          keys r<rid>_s<step>, float32 (3072,)
        embeddings/response/<task>.npz        same keys; observation embeddings

~25 GB total. After downloading into this directory:

    python scripts/build_slim_caches.py
    python scripts/pool_traj_worker.py --convert
