"""TDP Agent for T2-Bench.

A LocalAgent subclass that generates N samples per turn via Vertex MaaS,
records all samples as TDP steps, and returns the chosen one to the
T2-Bench orchestrator.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from typing import List, Optional

from pydantic import BaseModel, Field

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

from .data_types import Tau2StepSample, Tau2TDPStep
from .vertex_chat_sampler import generate_n_chat_samples as _vertex_generate
from .openai_chat_sampler import generate_n_chat_samples as _openai_generate

logger = logging.getLogger(__name__)


def _is_vertex_model(model: str) -> bool:
    """Return True if *model* should be routed through Vertex MaaS."""
    try:
        from .vertex_chat_sampler import vertex_maas_models
        if model in vertex_maas_models:
            return True
    except Exception:
        pass
    return False


_TDP_DEBUG = os.environ.get("TDP_DEBUG", "").lower() in ("1", "true", "yes")


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if not _TDP_DEBUG:
        return
    sys.stderr.write(f"[DEBUG] {message} | {json.dumps(data, default=str)}\n")
    sys.stderr.flush()

def _split_thinking_logprobs(
    logprobs: list[dict],
    reasoning_content: str,
    visible_content: str,
) -> tuple[list[dict], list[dict]]:
    """Split a logprob stream into reasoning and action portions.

    The thinking model returns reasoning_content separately but logprobs
    cover the full generation (thinking + action).  We find where the
    visible content begins in the token stream to split them.
    """
    token_stream = "".join(lp["token"] for lp in logprobs)

    if visible_content:
        prefix = visible_content[:60]
        split_pos = token_stream.find(prefix)
    else:
        split_pos = -1

    if split_pos <= 0:
        return logprobs, []

    cumul = ""
    split_idx = len(logprobs)
    for i, lp in enumerate(logprobs):
        cumul += lp["token"]
        if len(cumul) >= split_pos:
            split_idx = i
            break

    return logprobs[:split_idx], logprobs[split_idx:]


AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


# -- Message conversion helpers (tau2 types -> OpenAI dicts) ---------------

def _tau2_messages_to_openai(messages: list[Message]) -> list[dict]:
    """Convert tau2 message objects to OpenAI-format dicts for the API."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, UserMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AssistantMessage):
            d: dict = {"role": "assistant", "content": msg.content}
            if msg.is_tool_call():
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in msg.tool_calls
                ]
            result.append(d)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.id,
            })
    return result


def _raw_sample_to_assistant_message(sample: dict) -> AssistantMessage:
    """Convert a raw API sample dict into a tau2 AssistantMessage."""
    content = sample.get("content")
    raw_tcs = sample.get("tool_calls")
    tool_calls = None
    if raw_tcs:
        tool_calls = []
        for tc in raw_tcs:
            args = tc["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc["name"],
                arguments=args,
            ))
    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )


# -- TDP Agent state and class --------------------------------------------

class TDPAgentState(BaseModel):
    """Agent state that also accumulates TDP step data."""

    model_config = {"arbitrary_types_allowed": True}

    system_messages: list[SystemMessage] = Field(default_factory=list)
    messages: list[APICompatibleMessage] = Field(default_factory=list)
    tdp_steps: list[Tau2TDPStep] = Field(default_factory=list)
    step_counter: int = 0


class TDPAgent(LocalAgent[TDPAgentState]):
    """Agent that samples N completions per turn for TDP analysis.

    Plugs into the T2-Bench Orchestrator as a standard LocalAgent, but
    internally calls Vertex MaaS with n=N to collect multiple samples
    at each agent turn.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        model: str,
        n_samples: int = 10,
        temperature: float = 0.8,
        selection_strategy: str = "random",
        cost_tracker=None,
        logprobs: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.selection_strategy = selection_strategy
        self._openai_tools = [t.openai_schema for t in tools]
        self._cost_tracker = cost_tracker
        self.logprobs = logprobs
        self._api_base = api_base
        self._api_key = api_key
        self._use_vertex = _is_vertex_model(model)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> TDPAgentState:
        if message_history is None:
            message_history = []
        return TDPAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history),
            tdp_steps=[],
            step_counter=0,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: TDPAgentState
    ) -> tuple[AssistantMessage, TDPAgentState]:
        # region agent log
        _debug_log(
            "H1",
            "tdp_agent.py:181",
            "agent_generate_next_message_enter",
            {
                "step_counter": state.step_counter,
                "incoming_type": type(message).__name__,
                "state_message_count": len(state.messages),
                "system_message_count": len(state.system_messages),
            },
        )
        # endregion
        # 1. Append incoming message(s) to history
        input_msg_dict = None
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            input_msg_dict = {
                "role": "tool",
                "content": [
                    {"id": tm.id, "content": tm.content} for tm in message.tool_messages
                ],
            }
        else:
            state.messages.append(message)
            input_msg_dict = {"role": message.role}
            if hasattr(message, "content") and message.content:
                input_msg_dict["content"] = message.content
            if hasattr(message, "id"):
                input_msg_dict["id"] = message.id

        # 2. Build OpenAI-format messages for the API call
        all_messages = list(state.system_messages) + list(state.messages)
        openai_messages = _tau2_messages_to_openai(all_messages)
        # region agent log
        _debug_log(
            "H2",
            "tdp_agent.py:202",
            "agent_pre_vertex_request",
            {
                "step_counter": state.step_counter,
                "openai_message_count": len(openai_messages),
                "roles": [m.get("role") for m in openai_messages],
                "system_chars": sum(len(m.get("content") or "") for m in openai_messages if m.get("role") == "system"),
                "last_user_chars": next((len((m.get("content") or "")) for m in reversed(openai_messages) if m.get("role") == "user"), 0),
                "tool_count": len(self._openai_tools),
                "tool_names": [t.get("function", {}).get("name") for t in self._openai_tools[:6]],
                "model": self.model,
                "n_samples": self.n_samples,
                "temperature": self.temperature,
            },
        )
        # endregion

        # 3. Generate N samples
        logger.info(
            f"TDPAgent step {state.step_counter}: generating {self.n_samples} samples "
            f"({'vertex' if self._use_vertex else 'openai-compat'})"
        )
        sampler_kwargs = dict(
            model=self.model,
            messages=openai_messages,
            tools=self._openai_tools if self._openai_tools else None,
            n=self.n_samples,
            temperature=self.temperature,
            logprobs=self.logprobs,
        )
        if self._use_vertex:
            raw_samples, usage = _vertex_generate(**sampler_kwargs)
        else:
            raw_samples, usage = _openai_generate(
                **sampler_kwargs,
                api_base=self._api_base,
                api_key=self._api_key,
            )
        if self._cost_tracker is not None:
            self._cost_tracker.record(
                self.model, usage["prompt_tokens"], usage["completion_tokens"],
            )
        # region agent log
        _debug_log(
            "H4",
            "tdp_agent.py:228",
            "agent_post_vertex_samples",
            {
                "step_counter": state.step_counter,
                "raw_sample_count": len(raw_samples),
                "usage": usage,
                "first_sample_has_content": bool(raw_samples and raw_samples[0].get("content")),
                "first_sample_tool_calls": 0 if not raw_samples or not raw_samples[0].get("tool_calls") else len(raw_samples[0]["tool_calls"]),
                "first_sample_logprobs": 0 if not raw_samples else len(raw_samples[0].get("logprobs") or []),
                "first_sample_finish_reason": "" if not raw_samples else raw_samples[0].get("finish_reason", ""),
            },
        )
        # endregion

        # 4. Parse into Tau2StepSample objects
        parsed_samples: list[Tau2StepSample] = []
        for raw in raw_samples:
            logprobs_detail = raw.get("logprobs") or []
            token_lps = [lp["logprob"] for lp in logprobs_detail] if logprobs_detail else []
            total_lp = sum(token_lps) if token_lps else 0.0

            reasoning_content = raw.get("reasoning_content")
            reasoning_lps = None
            action_lps = None
            if reasoning_content and logprobs_detail:
                reasoning_lps, action_lps = _split_thinking_logprobs(
                    logprobs_detail, reasoning_content, raw.get("content") or "",
                )

            parsed_samples.append(Tau2StepSample(
                content=raw.get("content"),
                tool_calls=raw.get("tool_calls"),
                token_log_probs=token_lps,
                total_log_prob=total_lp,
                is_chosen=False,
                finish_reason=raw.get("finish_reason", ""),
                logprobs=logprobs_detail if logprobs_detail else None,
                reasoning_content=reasoning_content,
                reasoning_logprobs=reasoning_lps,
                action_logprobs=action_lps,
            ))

        # 5. Select one sample
        chosen_idx = self._select_sample(parsed_samples)
        parsed_samples[chosen_idx].is_chosen = True

        # 6. Record TDP step
        tdp_step = Tau2TDPStep(
            step_index=state.step_counter,
            samples=parsed_samples,
            input_message=input_msg_dict,
        )
        state.tdp_steps.append(tdp_step)
        state.step_counter += 1

        # 7. Convert chosen sample to AssistantMessage and append to history
        chosen_raw = raw_samples[chosen_idx]
        assistant_msg = _raw_sample_to_assistant_message(chosen_raw)
        # region agent log
        _debug_log(
            "H4",
            "tdp_agent.py:274",
            "agent_chosen_sample_shape",
            {
                "step_counter": state.step_counter,
                "chosen_idx": chosen_idx,
                "assistant_content_present": bool(assistant_msg.content),
                "assistant_tool_call_count": 0 if not assistant_msg.tool_calls else len(assistant_msg.tool_calls),
                "chosen_finish_reason": chosen_raw.get("finish_reason", ""),
                "chosen_logprobs_len": len(chosen_raw.get("logprobs") or []),
            },
        )
        # endregion
        state.messages.append(assistant_msg)

        return assistant_msg, state

    def _select_sample(self, samples: list[Tau2StepSample]) -> int:
        if self.selection_strategy == "random":
            return random.randint(0, len(samples) - 1)
        elif self.selection_strategy == "max_logprob":
            return max(range(len(samples)), key=lambda i: samples[i].total_log_prob)
        else:
            return random.randint(0, len(samples) - 1)

    def set_seed(self, seed: int):
        random.seed(seed)
