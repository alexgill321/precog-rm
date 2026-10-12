# Precog-RM

Code and data release for *"Divergent Paths: Learning an Outcome-Predictive
Representation of LLM Agent Trajectories"* (Alexander Gill, Guram Gogia,
Francisco Guzmán). Paper: **arXiv link TBD**.

Precog-RM reframes agentic tool-use trajectories as paths through a shared
embedding space of actions and environment observations, meta-learns a
representation in which rollout success can be predicted few-shot across unseen
tasks and policy models, and uses that signal as an early-stopping controller
for GRPO post-training (up to 47.3% fewer training FLOPs on Tau2-Bench at
comparable performance).

**Every number in the paper is reproducible from this repository plus the data
release on HuggingFace.** `REPLICATION.md` maps each table, figure and quoted
number to a command, with two paths: the shipped per-task outputs (seconds, no
download) and a full recompute from the released trajectories, embeddings,
checkpoints and GRPO logs.

## Data release

| HuggingFace repo | Contents | Size |
|---|---|---|
| `agill32/precog-rm-embeddings` | step embeddings for the 9 (benchmark, policy) cells, tau2 rollouts, reasoning embeddings, replay embeddings, optional mmap caches | 25 GB (+13 GB replay, +10 GB caches) |
| `agill32/precog-rm-checkpoints` | 75 sweep checkpoints, design-study and prefix-trained encoders, the deployed GRPO controller | 0.5 GB |
| `agill32/precog-rm-grpo-logs` | the 21 official GRPO runs (metrics, full rollouts, support sets), replay results, base-model eval | 35 GB |
| `agill32/precog-rm-rollouts` | raw BFCLv3 and AgentClinic trajectories, no-tools ablation, final-diagnosis resamples, task lists | 5 GB |

```bash
pip install -r requirements.txt
python scripts/download.py --stage checkpoints caches      # enough for every evaluation
python scripts/download.py --all                           # everything
```

Layout of the data root: `data/README.md`. Set `PRECOG_DATA` to use another
location.

## Quick start

```bash
# shipped results, no download
python scripts/print_table1.py                 # Table 1
python scripts/aggregate_generalization.py     # Table 2, App. J
python scripts/table_grpo.py                   # Table 3, App. K
python scripts/design_study.py                 # App. C, F

# full pipeline (after download --stage embeddings checkpoints)
python scripts/build_slim_caches.py && python scripts/pool_traj_worker.py --convert
python scripts/merged_cv_table.py && python scripts/print_table1.py
python scripts/precog_sweep.py                 # 25 pools x 3 seeds
python scripts/aggregate_generalization.py
python scripts/support_sweep.py && python scripts/fig_fewshot_bars.py
python scripts/prefix_gating_eval.py --train && python scripts/fig_prefix_gated.py
```

`make help` lists every stage.

## Layout

```
scripts/        one driver per paper result (see REPLICATION.md); _paths.py resolves the data root
src/prm/        Precog-RM model + episodic training (train_precog.py), state clustering
                (paired_action_response.py, cluster_sequence.py, thought_dag.py), text extraction (embeddings.py)
src/tau2_train/ GRPO training loop (grpo.py, train.py), early-stopping controller (early_stopping.py,
                train_early_stop.py), configs/*.yaml for all 21 official runs
exp/results/protonet_ablations/   per-task outputs of our runs (the "shipped results" path)
logs/<run>/metrics.jsonl          per-batch training metrics of the 21 official runs
notebooks/data/tau2_es/           per-rollout token cache + tokenizer calibration (rebuilt by grpo_cache.py)
benchmark_refs/AgentClinic/       tool-reliance scenario lists (rebuilt by notools_analysis.py)
data/                             data root (download target) + layout docs + HF dataset cards
```

## Environment

Python 3.11+ (developed on 3.13), `pip install -r requirements.txt`. Apple MPS
or CUDA for meta-training. External, not bundled: **tau2-bench** (GRPO only;
importable as `tau2`), **Tinker SDK** (`tinker` + `tinker_cookbook`; GRPO only,
`train_early_stop.py` patches private SDK modules so pin the version you train
with), **OpenAI API** (`OPENAI_API_KEY`; only for re-embedding rollouts).

## Reproducibility notes

Task splits are fixed (`exp/results/protonet_ablations/task_splits.json`).
Cross-validation uses `KFold(3, shuffle=True, random_state=42)` throughout;
support draws are seeded per (task, k, draw); Precog-RM training seeds {0,1,2}
control initialisation and episode order; GRPO runs use seeds {1,2,3}. fastText
baselines train multi-threaded without a seed and reproduce to within ±0.01.
The FLOPs accounting of Table 3 / Appendix K is in `scripts/grpo_common.py`.

## Citation

```bibtex
@article{gill2026divergent,
  title   = {Divergent Paths: Learning an Outcome-Predictive Representation of LLM Agent Trajectories},
  author  = {Gill, Alexander and Gogia, Guram and Guzmán, Francisco},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). Benchmark data derive from tau2-bench (MIT), gorilla/BFCL
(Apache-2.0) and AgentClinic (MedQA-Extended); see their repositories for terms.
