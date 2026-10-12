"""Training configuration for tau2 GRPO on Tinker."""

from dataclasses import dataclass, field


TINKER_PRICE_PREFILL_PER_M = 0.13
TINKER_PRICE_SAMPLE_PER_M = 0.40
TINKER_PRICE_TRAIN_PER_M = 0.40


@dataclass
class CostTracker:
    """Running token/cost counters for a training run."""
    prefill_tokens: int = 0
    sample_tokens: int = 0
    train_tokens: int = 0

    @property
    def prefill_cost(self) -> float:
        return self.prefill_tokens / 1_000_000 * TINKER_PRICE_PREFILL_PER_M

    @property
    def sample_cost(self) -> float:
        return self.sample_tokens / 1_000_000 * TINKER_PRICE_SAMPLE_PER_M

    @property
    def train_cost(self) -> float:
        return self.train_tokens / 1_000_000 * TINKER_PRICE_TRAIN_PER_M

    @property
    def total_cost(self) -> float:
        return self.prefill_cost + self.sample_cost + self.train_cost

    def summary(self) -> str:
        return (
            f"prefill={self.prefill_tokens:,}tok (${self.prefill_cost:.4f}) | "
            f"sample={self.sample_tokens:,}tok (${self.sample_cost:.4f}) | "
            f"train={self.train_tokens:,}tok (${self.train_cost:.4f}) | "
            f"TOTAL=${self.total_cost:.4f}"
        )


@dataclass
class EarlyStopConfig:
    """ProtoNetPRM early stopping configuration."""
    checkpoint_path: str = ""
    threshold: float = 0.8
    embedding_model: str = "text-embedding-3-large"
    min_steps: int = 1
    min_completed: int = 4
    support_save_dir: str | None = None
    rolling_window: int | None = None
    min_class_support: int = 16
    permanent_support_dir: str | None = None
    n_permanent_per_task: int = 16


@dataclass
class TrainConfig:
    # Tinker service
    model_name: str = "Qwen/Qwen3-8B"
    base_url: str | None = None
    lora_rank: int = 32
    renderer_name: str = "qwen3"

    # GRPO hyperparameters
    learning_rate: float = 2e-5
    kl_penalty_coef: float = 0.0
    kl_discount_factor: float = 0.0
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, float] | None = None
    num_substeps: int = 2

    # Rollout settings
    group_size: int = 8
    batch_size: int = 4
    max_tokens: int = 1024
    temperature: float = 1.0
    max_turns: int = 100
    max_trajectory_tokens: int = 65536
    max_concurrent_rollouts: int = 64

    # tau2 environment
    domains: list[str] = field(default_factory=lambda: ["retail"])
    task_ids: list[str] | None = None
    task_split: str | None = "base"
    user_model: str = "gpt-4o-mini"

    # Training loop
    seed: int = 42
    task_seed: int | None = None
    num_batches: int = 30
    save_every: int = 20
    eval_every: int = 5
    eval_group_size: int = 3
    eval_task_split: str | None = "test"
    log_path: str = "logs/tau2_tinker"
    run_name: str | None = None
    compute_post_kl: bool = False
    save_rollouts: bool = True

    # Reward shaping
    use_action_reward: bool = False

    # Early stopping
    early_stop: EarlyStopConfig | None = None

    # GCS upload (rollouts + metrics)
    gcs_bucket: str | None = None
    gcs_prefix: str | None = None  # set to your bucket prefix to enable GCS upload
    gcs_project: str | None = None  # set to your GCP project to enable GCS upload
