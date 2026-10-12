# Precog-RM

Code release for *"Divergent Paths: Learning an Outcome-Predictive
Representation of LLM Agent Trajectories"*
(Alexander Gill, Guram Gogia, Francisco Guzman).
Paper: **arXiv link TBD**.

Precog-RM reframes agentic tool-use trajectories as paths through a shared
embedding space of actions and environment observations, meta-learns a
representation in which rollout success can be predicted few-shot across
unseen tasks and policy models, and uses that signal as an early-stopping
controller for GRPO post-training (up to 47.3% fewer training FLOPs on
Tau2-Bench at comparable performance).

**This repository is primarily the implementation codebase.** It contains the
complete pipeline behind the paper's main results — preprocessing the embedded
rollout corpora, meta-training Precog-RM, evaluating it cross-task,
cross-model, and cross-benchmark, and running GRPO with the early-stopping
controller — such that every pipeline runs end-to-end **once the data
directories exist** (see "Running the pipelines" below for exactly how).
It also includes the raw per-task evaluation outputs of our runs, so the
paper's aggregations (Tables 1–3, Figures 3–5) can be recomputed and
inspected immediately, without any data.

## Data release

The pipelines consume ~25 GB of embedded rollout corpora (nine
(benchmark, model) cells of per-step `text-embedding-3-large` vectors), ~430 MB
of trained checkpoints, and ~56 GB of raw GRPO rollout logs. These are too
large for this repository and will be published separately (HuggingFace link
TBD). `data/*/PLACEHOLDER.md` specifies the exact directory layouts this code
expects, so the data release drops into place without code changes.

## What runs right now (no data needed)

The raw per-task outputs of our runs are included, and every main-result
aggregation is *computed* from them by shipped code (see `REPLICATION.md` for
details and expected outputs):

```
python scripts/print_table1.py               # Table 1 from per-task, per-fold AUROCs
python scripts/aggregate_generalization.py   # Table 2 (+ appendix matrix) from the
                                             #   75 sweep runs' per-epoch eval curves
python scripts/fig_grpo_curves.py            # Table 3 + Figure 5 from raw batch
                                             #   metrics and per-rollout token counts
notebooks/fig_fewshot_bars.ipynb             # Figure 3 from per-task sweep outputs
notebooks/fig_prefix_gated.ipynb             # Figure 4 from per-task gating outputs
```

## Running the pipelines (with the data in place)

**1. Preprocessing.** Place the nine embedding corpora under
`data/table1_data/<benchmark>_<model>/` (layout in
`data/table1_data/PLACEHOLDER.md`), then:
```
python scripts/build_slim_caches.py          # per-task clustering + compact caches
python scripts/pool_traj_worker.py --convert # memory-mapped caches used by everything below
```
This reproduces the full curation protocol of Appendix D (mixed-outcome
filtering, AgentClinic final-step drop, per-task HDBSCAN state clustering).

**2. Meta-train Precog-RM.** The 25-pool × 3-seed sweep of the paper:
```
python scripts/precog_sweep.py
```
Trains every pool composition (single-model, two-model, benchmark specialist,
two-benchmark, generalist; 10,000 episodes each), writes checkpoints and the
per-epoch held-out evaluation grids
(`exp/results/protonet_ablations/precog_parts/`). `scripts/pool_traj_worker.py`
also exposes the ablation registry of Appendices C/F (`--ablation`).

**3. Cross-task / cross-model / cross-benchmark evaluation.** Aggregate the
sweep into the paper's generalization matrices:
```
python scripts/aggregate_generalization.py
```

**4. Matched-supervision comparison (Table 1).** Per-task discriminators
(NC, RBF, fastText over state paths) against Precog-RM on identical CV folds:
```
python scripts/merged_cv_table.py            # needs data/ckpts (or step 2's output)
python scripts/print_table1.py
```

**5. Few-shot evaluation (Figure 3).**
```
python scripts/support_sweep.py              # identical seeded draws, k in {1,2,4,8}
```

**6. GRPO with early stopping (Table 3).** Requires the Tinker SDK and
`TINKER_API_KEY`; tau2-bench importable as `tau2`:
```
python -m src.tau2_train.train            --config src/tau2_train/configs/baseline_retail_official_s1.yaml
python -m src.tau2_train.train_early_stop --config src/tau2_train/configs/early_stop_retail_official_s1.yaml
python scripts/fig_grpo_curves.py            # FLOPs accounting over the runs' logs
```
The 21 official runs of Table 3 are the `*_official_*` configs, seeds s1–s3;
all Tinker run configs used in the project are included.

**7. Prefix prediction and confidence gating (Figure 4, Section 4.5).**
```
python scripts/prefix_gating_eval.py         # score every prefix of every held-out
                                             #   rollout; fraction curves + gated frontier
```
Loads (or with `--train`, meta-trains) the prefix-trained variants, writes
per-cell results into `exp/results/protonet_ablations/prefix_results.json`,
which `notebooks/fig_prefix_gated.ipynb` plots.

## Layout

```
scripts/                     pipeline drivers (see above) + shared protocol
                             (pool_traj_worker.py: loaders, pools, eval rules)
src/prm/                     Precog-RM model + episodic training (train_precog.py),
                             state clustering (paired_action_response.py,
                             cluster_sequence.py), action/observation text
                             extraction (embeddings.py)
src/tau2_train/              GRPO training loop (grpo.py, train.py), the
                             early-stopping controller (early_stopping.py,
                             train_early_stop.py), configs/*.yaml
exp/results/protonet_ablations/  raw per-task outputs of our runs (see REPLICATION.md)
notebooks/data/tau2_es/      GRPO per-rollout token cache + $/token calibration
logs/<run>/metrics.jsonl     per-batch training metrics, all 21 official runs
benchmark_refs/AgentClinic/  tool-reliance scenario ID lists (Appendix D)
data/                        expected layouts for the published data
```

## Environment

Python 3.11+ (developed on 3.13). `pip install -r requirements.txt`.
External, not bundled: **tau2-bench** (GRPO only; importable as `tau2`),
**Tinker SDK** (`tinker` + `tinker_cookbook`; GRPO only — note
`train_early_stop.py` patches private SDK modules, pin the version you train
with), **OpenAI API** (`OPENAI_API_KEY`; only for re-embedding rollouts).

## Reproducibility notes

Task splits are fixed (`exp/results/protonet_ablations/task_splits.json`).
Cross-validation uses `KFold(3, shuffle=True, random_state=42)` throughout;
support draws are seeded per (task, k, draw); Precog-RM training seeds {0,1,2}
control initialization and episode order; GRPO runs use seeds {1,2,3}. The
FLOPs accounting of Table 3 / Appendix K is implemented in
`scripts/fig_grpo_curves.py`.

## Citation

```bibtex
@article{gill2026divergent,
  title   = {Divergent Paths: Learning an Outcome-Predictive Representation of LLM Agent Trajectories},
  author  = {Gill, Alexander and Gogia, Guram and Guzman, Francisco},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT — see `LICENSE`.
