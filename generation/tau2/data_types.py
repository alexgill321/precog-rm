"""Core data structures for T2-Bench TDP sampling.

Adapted from src/uprop/data_types.py for chat-based tool-calling
interactions instead of ReAct-style text loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tau2StepSample:
    """One of N sampled continuations at a single agent turn."""

    content: Optional[str]
    tool_calls: Optional[list[dict]]
    token_log_probs: list[float]
    total_log_prob: float
    is_chosen: bool
    finish_reason: str = ""
    logprobs: Optional[list[dict]] = None
    reasoning_content: Optional[str] = None
    reasoning_logprobs: Optional[list[dict]] = None
    action_logprobs: Optional[list[dict]] = None

    def to_dict(self) -> dict:
        d = {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "token_log_probs": self.token_log_probs,
            "total_log_prob": self.total_log_prob,
            "is_chosen": self.is_chosen,
            "finish_reason": self.finish_reason,
            "logprobs": self.logprobs,
        }
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
            d["reasoning_logprobs"] = self.reasoning_logprobs
            d["action_logprobs"] = self.action_logprobs
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Tau2StepSample:
        return cls(
            content=d.get("content"),
            tool_calls=d.get("tool_calls"),
            token_log_probs=d.get("token_log_probs", []),
            total_log_prob=d.get("total_log_prob", 0.0),
            is_chosen=d.get("is_chosen", False),
            finish_reason=d.get("finish_reason", ""),
            logprobs=d.get("logprobs"),
            reasoning_content=d.get("reasoning_content"),
            reasoning_logprobs=d.get("reasoning_logprobs"),
            action_logprobs=d.get("action_logprobs"),
        )


@dataclass
class Tau2TDPStep:
    """All N samples at one agent turn, plus the context that triggered it."""

    step_index: int
    samples: list[Tau2StepSample]
    input_message: Optional[dict] = None
    env_responses: Optional[list[dict]] = None

    @property
    def chosen(self) -> Tau2StepSample:
        return next(s for s in self.samples if s.is_chosen)

    @property
    def chosen_index(self) -> int:
        return next(i for i, s in enumerate(self.samples) if s.is_chosen)

    @property
    def alternatives(self) -> list[Tau2StepSample]:
        return [s for s in self.samples if not s.is_chosen]

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "samples": [s.to_dict() for s in self.samples],
            "input_message": self.input_message,
            "env_responses": self.env_responses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Tau2TDPStep:
        return cls(
            step_index=d["step_index"],
            samples=[Tau2StepSample.from_dict(s) for s in d["samples"]],
            input_message=d.get("input_message"),
            env_responses=d.get("env_responses"),
        )


@dataclass
class Tau2TDP:
    """One Trajectory-Dependent Decision Process for a T2-Bench task."""

    task_id: str
    model_id: str
    tdp_index: int
    domain: str
    steps: list[Tau2TDPStep]
    messages: list[dict] = field(default_factory=list)
    reward: Optional[float] = None
    reward_info: Optional[dict] = None
    termination_reason: str = ""
    seed: Optional[int] = None
    cost_usd: Optional[float] = None

    @property
    def trajectory_length(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "tdp_index": self.tdp_index,
            "domain": self.domain,
            "steps": [s.to_dict() for s in self.steps],
            "messages": self.messages,
            "reward": self.reward,
            "reward_info": self.reward_info,
            "termination_reason": self.termination_reason,
            "seed": self.seed,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Tau2TDP:
        return cls(
            task_id=d["task_id"],
            model_id=d["model_id"],
            tdp_index=d["tdp_index"],
            domain=d.get("domain", ""),
            steps=[Tau2TDPStep.from_dict(s) for s in d["steps"]],
            messages=d.get("messages", []),
            reward=d.get("reward"),
            reward_info=d.get("reward_info"),
            termination_reason=d.get("termination_reason", ""),
            seed=d.get("seed"),
            cost_usd=d.get("cost_usd"),
        )
