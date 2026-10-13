# Replication guide

Every number, table and figure in the paper maps to one command below. Two
entry points:

* **(A) shipped results** — the per-task outputs of our runs are in the repo
  (`exp/results/protonet_ablations/*.json`, `logs/*/metrics.jsonl`,
  `notebooks/data/tau2_es/`). The aggregation commands run in seconds with no
  download and reproduce the printed values.
* **(B) full recompute** — download the data release (`scripts/download.py`),
  rebuild the caches, retrain, re-evaluate. Expected run times are for an
  Apple M-series laptop (MPS) unless noted.

Set `PRECOG_DATA` to the data root (default `./data`); see `data/README.md`.
Values marked *exact* reproduce the shipped JSON bit-for-bit under (B); values
marked *±0.01* depend on fastText, which trains multi-threaded without a seed.

```
python scripts/download.py --part checkpoints caches        # 11 GB: everything except retraining + raw data
python scripts/download.py --part embeddings                 # 38 GB: raw embeddings (rebuild caches yourself)
python scripts/download.py --part grpo rollouts              # 40 GB: GRPO logs, raw trajectories
```

---

## Preprocessing (B only)

| Step | Command | Time |
|---|---|---|
| per-task clustering + slim caches | `python scripts/build_slim_caches.py` | ~1.5 h, 9 cells |
| memory-mapped caches | `python scripts/pool_traj_worker.py --convert` | minutes |
| dataset statistics (Sec. 4.1, App. D, tab:pool-sizes) | `python scripts/dataset_stats.py` | seconds |

`caches/task_splits.json` (the fixed one-third held-out split, shared across
policies) is shipped and must not be regenerated.

## Main text

| Result | (A) shipped | (B) recompute | Status |
|---|---|---|---|
| Fig. 1 trajectory flow (tau2/qwen `retail/103`) | — | `python scripts/fig_trajectory_flow.py` | content identical; paper PDF used a hand-tuned renderer with a token axis |
| Table 1 separability (9 cells x NC/RBF/FT/Precog) | `python scripts/print_table1.py` | `python scripts/merged_cv_table.py` (specialists + generalist, ~2 h) then `print_table1.py` | exact (A); FT columns ±0.01 (B) |
| Fig. 3 few-shot bars + App. H table | `python scripts/print_fewshot_table.py; python scripts/fig_fewshot_bars.py` | `python scripts/support_sweep.py` (~1 h) | exact (A) |
| Table 2 cross-model generalization + OOD means | `python scripts/aggregate_generalization.py --check` | `python scripts/precog_sweep.py` (75 runs x 10k episodes, ~40 GPU-h) | exact (A) |
| Sec. 4.5 gating operating points + Fig. 4 | `python scripts/fig_prefix_gated.py` | `python scripts/prefix_gating_eval.py --train` (~3 h) | exact (A) |
| Table 3 FLOPs saved / pass@1, retained gain; Sec. 5.2 reward trends | `python scripts/table_grpo.py` | `python scripts/grpo_cache.py --rebuild` from `runs/` (verified identical), then `table_grpo.py` | exact |
| Fig. 5 training curves | `python scripts/fig_grpo_curves.py` | same | exact |
| GRPO training itself (21 runs) | — | `python -m src.tau2_train.train --config src/tau2_train/configs/baseline_retail_official_s1.yaml`; `python -m src.tau2_train.train_early_stop --config src/tau2_train/configs/early_stop_retail_official_s1.yaml` (Tinker SDK + tau2-bench + `OPENAI_API_KEY`; ~$150/run) | stochastic |

## Appendix

| Result | (A) shipped | (B) recompute | Status |
|---|---|---|---|
| App. B tab:channel (9 cells x 4 channels) | `python scripts/q4_channel_ablation.py --print-only` | `python scripts/q4_channel_ablation.py --force` (~1.5 h) | raw exact (all 36 values); clustered ±0.005 on Tau2/BFCL, ±0.02 on the 13-task AgentClinic cells |
| App. B.1 tab:reasoning | `python scripts/reasoning_signal_ablation.py` | `--embed` to re-embed reasoning (`OPENAI_API_KEY`, ~$6) | raw exact; clustered ±0.01 |
| App. C tab:ablations, mixed-pool replication | `python scripts/design_study.py` | `--eval` re-scores shipped checkpoints; `--train` retrains (~20 min each) | exact (A) |
| App. E tab:pool-sizes, App. D counts | `python scripts/dataset_stats.py` | same | exact |
| App. F tab:design-ablations, distance rule | `python scripts/design_study.py` (`--distance` to recompute) | | exact (A) |
| App. I no-tools counts, tool-reliant lists | `python scripts/notools_analysis.py --check` | regeneration: `benchmark_refs/AgentClinic/run_notools.py` (API) | exact |
| App. I tab:ac-resample | `python scripts/resample_summary.py` | regeneration: `benchmark_refs/AgentClinic/resample_diagnosis.py` (API) | exact |
| App. J tab:gen-benchmarks + untrained row | `python scripts/aggregate_generalization.py; python scripts/eval_untrained.py` | `eval_untrained.py --check` (recomputes; verified identical) | exact |
| App. K tab:flops-detail, 88/12 split, emb/gen % | `python scripts/table_grpo.py` | | exact |
| App. L tab:sep-shared | `python scripts/table_sep_shared.py` | after `merged_cv_table.py` | exact (A) |
| App. M Fig. 6 + quoted replay numbers | `python scripts/replay_sweep_table.py; python scripts/fig_sim_pareto.py` | `python scripts/embed_baseline_rollouts.py` (API) then `python scripts/run_grpo_early_stopping_official.py --checkpoint $PRECOG_DATA/ckpts/prefix/grpo_controller_model_prefix.pt --seed 1 2 3` (~1 h) | exact |
| GRPO controller checkpoint | shipped (`ckpts/prefix/grpo_controller_model_prefix.pt`) | `python scripts/run_prefix_verify.py --domains retail` | stochastic |

## Known differences from the printed paper

* Table 1 cells differ from the printed values by at most 0.001 (rounding of the
  fold-level aggregation); `print_table1.py` is the reference.
* App. K text: the controller's cost at policy parity ranges 0.8 to 2.3
  points across configurations (paper says 0.8 to 1.5).
* App. M text: permanent-support reward bias at theta=0.9 is 0.027 (paper says
  below 0.02 for both strategies).
* Sec. 4.5 text: BFCLv3 stop rate at theta=0.8 is 56%, printed as 59%.
* AgentClinic resampling for Qwen covers 19 scenarios / 190 rollouts (one
  scenario lacked a usable diagnosis prefix), printed as 20 x 10.

## Provenance of shipped caches

| Artifact | Producer |
|---|---|
| `exp/results/protonet_ablations/precog_parts/*.json` | `scripts/precog_sweep.py` |
| `merged_cv_table{,_spec}.json` | `scripts/merged_cv_table.py` |
| `support_sweep.json` | `scripts/support_sweep.py` |
| `prefix_results.json` | `scripts/prefix_gating_eval.py` |
| `results.json`, `disttest_results.json` | `scripts/design_study.py` |
| `untrained_precog.json` | `scripts/eval_untrained.py` |
| `channel_ablation.json`, `reasoning_ablation.json` | `scripts/q4_channel_ablation.py`, `scripts/reasoning_signal_ablation.py` |
| `notebooks/data/tau2_es/rollout_cache/*.parquet`, `calibration.json` | `scripts/grpo_cache.py` (tokenizer calibration from the notebook; constant) |
| `notebooks/data/tau2_es/replay_threshold_sweep.csv` | `scripts/replay_sweep_table.py` |
| `logs/<run>/metrics.jsonl` | written per batch by `src/tau2_train/train*.py` |
| `benchmark_refs/AgentClinic/*.txt` | `scripts/notools_analysis.py` |
