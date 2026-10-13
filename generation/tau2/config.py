"""Configuration for T2-Bench TDP sampling experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Tau2TDPConfig:
    """Hyperparameters and paths for T2-Bench TDP sampling."""

    # TDP sampling
    Z: int = 10
    N: int = 1
    temperature: float = 0.8
    max_steps: int = 100
    max_errors: int = 10

    # Model for agent generation (Vertex MaaS key or HF model ID for vLLM)
    model: str = "qwen3-235b"

    # OpenAI-compatible endpoint for non-Vertex models (env: OPENAI_COMPAT_BASE_URL)
    api_base: str | None = None
    # API key for the endpoint (env: OPENAI_COMPAT_API_KEY)
    api_key: str | None = None

    # Model for T2-Bench user simulator (uses LiteLLM)
    user_model: str = "gpt-4o-mini"

    # T2-Bench domain and task configuration
    domain: str = "airline"
    task_split: str = "base"
    task_ids: list[str] | None = None
    num_tasks: int | None = None

    # Selection strategy for choosing among N samples
    selection_strategy: str = "random"

    seed: int = 42
    max_concurrency: int = 1

    output_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "..", "..", "log", "tau2_tdp",
    ))

    def resolve_paths(self) -> None:
        self.output_dir = os.path.realpath(self.output_dir)
