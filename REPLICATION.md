# Replication guide

Every artifact has two paths: **(A) rebuild from the caches included in this
release** (no downloads, seconds-to-minutes) and **(B) full recompute from the
HuggingFace data release** (see `data/*/PLACEHOLDER.md` for layouts).

Run everything from the release root.

---

## Table 2 — cross-model generalization (and the appendix benchmark matrix)

**(A) Included.**
```
python scripts/aggregate_generalization.py --check   # verify against shipped JSON
python scripts/aggregate_generalization.py           # regenerate + print matrices
```
Reads the sweep's per-epoch eval curves
(`exp/results/protonet_ablations/precog_parts/<pool>_s<seed>.json`, included)
and applies best-within-pool stopping. Prints both 7×3 matrices; Table 2 is
`models`, the appendix cross-benchmark table is `benchmarks`.

**(B) Full.** Rebuild the caches (below), then retrain the sweep:
```
python scripts/precog_sweep.py            # 25 pools x 3 seeds -> precog_parts/
python scripts/aggregate_generalization.py
```

## Table 3 — GRPO early stopping (and Figure 5)

**(A) Included.**
```
python scripts/fig_grpo_curves.py
```
Reads `logs/<run>/metrics.jsonl` (billed-cost deltas → trained/embedded tokens)
and `notebooks/data/tau2_es/rollout_cache/*.parquet` (exact generation token
counts; distilled from the raw logs, no text), applies the FLOPs accounting of
Appendix K (2N gen + 6N train + 2N embed at policy parity, N = 8.19e9), writes
Figure 5, and prints the per-configuration savings column of Table 3
(expected: 10.1±1.4 / 25.3±3.6 / 35.5±2.2 / 43.0±4.2 / 47.3±1.8 / 63.0±2.0).

**(B) Full.** Train the runs yourself:
```
export TINKER_API_KEY=...
python -m src.tau2_train.train            --config src/tau2_train/configs/baseline_retail_official_s1.yaml
python -m src.tau2_train.train_early_stop --config src/tau2_train/configs/early_stop_retail_official_s1.yaml
```
(21 official runs = the `*_official_*` configs, seeds s1–s3.) The per-rollout
token cache is then rebuilt from `logs/<run>/rollouts/batch_*.jsonl` — the raw
logs ship in the data release (`data/grpo_rollouts/`).

## Table 1 — separability + Precog-RM at matched CV

**(A) Included.**
```
python scripts/print_table1.py
```
Computes the full table (fold-level aggregation, seed-averaged Precog-RM) from
the released per-task, per-fold raw outputs
(`merged_cv_table.json` for the baselines, `merged_cv_table_spec.json` for the
per-benchmark specialist Precog-RM reported in the paper). Values match the
printed table within ±0.001 (fold-aggregation rounding).

**(B) Full.** Needs the embedding corpora and checkpoints from the data
release:
```
python scripts/build_slim_caches.py        # data/table1_data/* -> slim caches
python scripts/pool_traj_worker.py --convert   # slim -> memory-mapped caches
python scripts/merged_cv_table.py          # requires data/ckpts checkpoints
```

## Figure 3 — few-shot support sweep

**(A) Included.** `exp/results/protonet_ablations/support_sweep.json` +
`notebooks/fig_fewshot_bars.ipynb` (run all cells; writes
`paper/figures/fig_fewshot_bars.pdf`).

**(B) Full.** After (B) of Table 1: `python scripts/support_sweep.py`.

## Figure 4 — prefix confidence gating

**(A) Included.** `exp/results/protonet_ablations/prefix_results.json` +
`notebooks/fig_prefix_gated.ipynb` (run all cells).

**(B) Full.** After the caches and checkpoints exist:
```
python scripts/prefix_gating_eval.py         # add --train to meta-train the
                                             #   prefix variants from scratch
```
(Extraction verified: recomputing a full evaluation cell with this script
reproduces the shipped `prefix_results.json` values exactly.)

---

### Provenance of included caches

| Artifact | Producer |
|---|---|
| `precog_parts/*.json` | `scripts/precog_sweep.py` (25 pools × 3 seeds, 10k episodes each) |
| `rollout_cache/*.parquet`, `calibration.json` | distilled from raw GRPO logs (token counts and tokenizer-calibrated $/token; no text) |
| `logs/<run>/metrics.jsonl` | written per batch by `src/tau2_train/train*.py` during the official runs |
| `merged_cv_table{,_spec}.json` | `scripts/merged_cv_table.py` |
| `support_sweep.json` | `scripts/support_sweep.py` |
| `prefix_results.json` | prefix-scoring evaluation of the prefix-trained checkpoints |
| `task_splits.json` | fixed held-out task split (1/3 per benchmark, shared across policies) |
| `benchmark_refs/AgentClinic/*.txt` | tool-reliance scenario labels (Appendix D filter) |
