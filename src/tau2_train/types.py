"""Data types for the GRPO training loop.

These mirror aurum's types so that tinker_cookbook utilities
(compute_advantages, assemble_training_data, etc.) work via duck typing.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TrajectoryID:
    instance_id: str
    repetition_id: int


@dataclass
class TokensWithLogprobs:
    tokens: list[int]
    maybe_logprobs: list[float] | None

    @property
    def logprobs(self) -> list[float]:
        if self.maybe_logprobs is None:
            raise ValueError("Logprobs are not available")
        return self.maybe_logprobs


@dataclass
class Transition:
    ob: list[dict] | object  # list[dict] before rendering, ModelInput after
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: dict[str, float | int] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: TrajectoryID
    transitions: list[Transition]
    final_ob: list[dict] | object

    def get_total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)


@dataclass
class TrajectoryGroup:
    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]
    metrics_G: Optional[list[dict]] = None

    def get_total_rewards(self) -> list[float]:
        return self.final_rewards_G
