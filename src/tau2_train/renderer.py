"""Chat template renderers for Tinker model input.

Renders conversations into token sequences matching Qwen3's native chat
template, including tool calling format and thinking-disabled mode.
"""

from __future__ import annotations

import json
import logging
import re
from typing import NotRequired, TypedDict

import tinker
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

Tokenizer = PreTrainedTokenizer

Role = str


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, str]


class Message(TypedDict):
    role: Role
    content: str
    tool_calls: NotRequired[list[ToolCall]]
    thinking: NotRequired[str]
    trainable: NotRequired[bool]


class Renderer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        raise NotImplementedError

    def get_stop_sequences(self) -> list[str] | list[int]:
        raise NotImplementedError

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        raise NotImplementedError

    def encode_new_turn(self, observation: str) -> list[int]:
        raise NotImplementedError


def parse_response_for_stop_token(
    response: list[int], tokenizer: Tokenizer, stop_token: int
) -> tuple[Message, bool]:
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = tokenizer.decode(response)
        logger.debug(f"Response missing stop token: {str_response[:200]}")
        return Message(role="assistant", content=str_response), False
    elif emt_count == 1:
        str_response = tokenizer.decode(response[: response.index(stop_token)])
        return Message(role="assistant", content=str_response), True
    else:
        raise ValueError(
            f"Expected 0 or 1 stop tokens in response, got {emt_count}. "
            "Check that the correct stop tokens are configured for sampling."
        )


class Qwen3Renderer(Renderer):
    """Qwen3 chat template renderer matching the model's native format.

    Token format (with tools, thinking disabled):

        <|im_start|>system
        {system_content}

        # Tools
        ...tool schemas...
        <|im_end|>
        <|im_start|>user
        ...<|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        ...<|im_end|>

    Tool responses are wrapped in <tool_response> tags inside user messages,
    matching Qwen3's chat template exactly.
    """

    THINKING_DISABLED_PREFILL = "<think>\n\n</think>\n\n"

    def _render_message(
        self, idx: int, message: Message
    ) -> tuple[list[int], list[int], list[int]]:
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message.get("content", "") or ""
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                tc_json = json.dumps({"name": tc["name"], "arguments": tc["arguments"]})
                if ac_content:
                    ac_content += "\n"
                ac_content += f"<tool_call>\n{tc_json}\n</tool_call>"
        ac_content += "<|im_end|>"
        ac_tail_str = ""
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_content, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        tokens: list[int] = []
        for idx, message in enumerate(messages):
            ob_part, action_part, _ = self._render_message(idx, message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        new_partial_message = Message(role=role, content="")
        ob_part, _, _ = self._render_message(len(messages), new_partial_message)
        tokens.extend(ob_part)
        thinking_prefix = self.THINKING_DISABLED_PREFILL
        if prefill:
            thinking_prefix += prefill
        tokens.extend(self.tokenizer.encode(thinking_prefix, add_special_tokens=False))
        return tinker.ModelInput.from_ints(tokens)

    def encode_new_turn(self, observation: str) -> list[int]:
        """Encode the tokens for a new user observation + assistant prefill.

        Produces the token sequence:
            \n<|im_start|>user\n{observation}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n

        This is meant to be appended to running_tokens (which already end
        with the sampled response tokens including <|im_end|>) so that the
        next sampling call has an exact token-level continuation.
        """
        user_turn = f"\n<|im_start|>user\n{observation}<|im_end|>"
        assistant_start = f"\n<|im_start|>assistant\n{self.THINKING_DISABLED_PREFILL}"
        text = user_turn + assistant_start
        return self.tokenizer.encode(text, add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_tool_call(self, tool_call_str: str) -> list[ToolCall] | None:
        try:
            tool_call = json.loads(tool_call_str)
        except json.JSONDecodeError:
            return None
        if not isinstance(tool_call, dict):
            return None
        if "name" not in tool_call:
            return None
        args = tool_call.get("arguments", tool_call.get("args", None))
        if args is None:
            return None
        return [ToolCall(name=tool_call["name"], arguments=args)]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        match = re.search(
            r"<tool_call>(.*?)</tool_call>", assistant_message["content"], re.DOTALL
        )
        if match:
            tool_calls = self._parse_tool_call(match.group(1))
            if tool_calls is None:
                return assistant_message, False
            assistant_message["tool_calls"] = tool_calls
            return assistant_message, True
        return assistant_message, True


def get_renderer(name: str, tokenizer: Tokenizer) -> Renderer:
    if name == "qwen3":
        return Qwen3Renderer(tokenizer)
    else:
        raise ValueError(
            f"Unknown renderer: {name!r}. "
            "Add more renderers here or import from aurum if needed."
        )
