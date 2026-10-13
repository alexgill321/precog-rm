PY ?= python
export PRECOG_DATA ?= $(CURDIR)/data

.PHONY: help download-eval download-all caches shipped table1 sweep fewshot prefix design signal agentclinic grpo replay figures all

help:
	@echo "Stages (see REPLICATION.md):"
	@echo "  download-eval   checkpoints + mmap caches (11 GB): enough to re-evaluate everything"
	@echo "  download-all    every HF stage (~85 GB)"
	@echo "  shipped         print every table from the shipped per-task outputs (no download)"
	@echo "  caches          rebuild slim + mmap caches from table1_data"
	@echo "  table1 sweep fewshot prefix design signal agentclinic grpo replay figures"
	@echo "  all             everything above that does not retrain (needs download-eval + grpo + rollouts)"

download-eval:
	$(PY) scripts/download.py --part checkpoints caches
download-all:
	$(PY) scripts/download.py --all

shipped:
	$(PY) scripts/print_table1.py
	$(PY) scripts/aggregate_generalization.py --check
	$(PY) scripts/print_fewshot_table.py
	$(PY) scripts/table_grpo.py
	$(PY) scripts/design_study.py
	$(PY) scripts/table_sep_shared.py
	$(PY) scripts/q4_channel_ablation.py --print-only
	$(PY) scripts/reasoning_signal_ablation.py --print-only
	$(PY) scripts/eval_untrained.py

caches:
	$(PY) scripts/build_slim_caches.py
	$(PY) scripts/pool_traj_worker.py --convert
	$(PY) scripts/dataset_stats.py

table1:
	$(PY) scripts/merged_cv_table.py
	$(PY) scripts/print_table1.py
	$(PY) scripts/table_sep_shared.py

sweep:
	$(PY) scripts/precog_sweep.py
	$(PY) scripts/aggregate_generalization.py
	$(PY) scripts/eval_untrained.py --force

fewshot:
	$(PY) scripts/support_sweep.py
	$(PY) scripts/print_fewshot_table.py
	$(PY) scripts/fig_fewshot_bars.py

prefix:
	$(PY) scripts/prefix_gating_eval.py --train
	$(PY) scripts/fig_prefix_gated.py

design:
	$(PY) scripts/design_study.py --eval --distance

signal:
	$(PY) scripts/q4_channel_ablation.py --force
	$(PY) scripts/reasoning_signal_ablation.py

agentclinic:
	$(PY) scripts/notools_analysis.py --check
	$(PY) scripts/resample_summary.py

grpo:
	$(PY) scripts/grpo_cache.py --rebuild
	$(PY) scripts/table_grpo.py
	$(PY) scripts/fig_grpo_curves.py

replay:
	$(PY) scripts/run_grpo_early_stopping_official.py --checkpoint $(PRECOG_DATA)/ckpts/prefix/grpo_controller_model_prefix.pt --seed 1 2 3
	$(PY) scripts/replay_sweep_table.py
	$(PY) scripts/fig_sim_pareto.py

figures:
	$(PY) scripts/fig_trajectory_flow.py
	$(PY) scripts/fig_fewshot_bars.py
	$(PY) scripts/fig_prefix_gated.py
	$(PY) scripts/fig_grpo_curves.py
	$(PY) scripts/replay_sweep_table.py && $(PY) scripts/fig_sim_pareto.py

all: caches table1 fewshot design signal agentclinic grpo figures
