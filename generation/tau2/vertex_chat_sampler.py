"""N-sample chat completion with tool calling via Vertex MaaS.

Extends the pattern from src/uprop/tdp_sampler.py to handle OpenAI-style
tool-calling chat completions instead of plain text completions.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from functools import wraps

logger = logging.getLogger(__name__)


_TDP_DEBUG = os.environ.get("TDP_DEBUG", "").lower() in ("1", "true", "yes")


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if not _TDP_DEBUG:
        return
    sys.stderr.write(f"[DEBUG] {message} | {json.dumps(data, default=str)}\n")
    sys.stderr.flush()

from .vertex_maas import (
    vertex_maas_models,
    vertex_maas_generation_defaults,
    _get_vertex_access_token,
    _get_vertex_project_id,
    _build_vertex_base_url,
    retry_with_backoff,
)


import re

_HARMONY_TAG_RE = re.compile(r"<\|(?:start|end|channel|message|call|return|constrain)\|>")
_COMMENTARY_TC_RE = re.compile(
    r"<\|start\|>assistant<\|channel\|>commentary\s+to=functions\.(\S+)\s*"
    r"<\|constrain\|>json<\|message\|>(.*?)<\|call\|>",
    re.DOTALL,
)


def _sanitize_gptoss_content(content: str | None, existing_tool_calls: list[dict] | None):
    """Strip raw gpt-oss Harmony tags from content.

    Vertex MaaS occasionally fails to parse gpt-oss Harmony output, leaking
    raw control tokens (``<|channel|>``, ``<|start|>``, etc.) into
    ``message.content``.  If left unsanitised these tokens cascade into the
    conversation history and corrupt all subsequent model replies.

    Returns ``(clean_content, tool_calls)`` where *tool_calls* is either the
    original ``existing_tool_calls`` or a list extracted from a ``commentary
    to=functions.*`` block when the server failed to populate ``tool_calls``.
    """
    if not content or not _HARMONY_TAG_RE.search(content):
        return content, existing_tool_calls

    # Extract the user-facing text: everything before the first control token.
    first_tag = _HARMONY_TAG_RE.search(content)
    clean = content[:first_tag.start()].strip() if first_tag else content.strip()

    # If the server didn't populate tool_calls, try to recover from commentary.
    tool_calls = existing_tool_calls
    if not tool_calls:
        match = _COMMENTARY_TC_RE.search(content)
        if match:
            fn_name = match.group(1)
            raw_args = match.group(2).strip()
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = raw_args
            tool_calls = [{
                "id": f"recovered_{fn_name}",
                "name": fn_name,
                "arguments": args,
            }]
            if not clean:
                clean = None

    return clean or None, tool_calls


def _parse_logprobs_from_choice(choice) -> list[dict]:
    """Extract parsed logprobs list from an OpenAI chat completion choice."""
    parsed = []
    if choice.logprobs:
        content_lps = getattr(choice.logprobs, "content", None)
        if content_lps:
            for tlp in content_lps:
                top = []
                for alt in getattr(tlp, "top_logprobs", []) or []:
                    top.append({"token": alt.token, "logprob": alt.logprob})
                parsed.append({
                    "token": tlp.token,
                    "logprob": tlp.logprob,
                    "top_logprobs": top,
                })
    return parsed


def _parse_tool_calls_from_choice(choice) -> list[dict] | None:
    """Extract tool calls from an OpenAI chat completion choice."""
    raw_tc = choice.message.tool_calls
    if not raw_tc:
        return None
    result = []
    for tc in raw_tc:
        args = tc.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        result.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": args,
        })
    return result


def _build_client(model: str):
    """Create an OpenAI client pointed at the Vertex MaaS endpoint."""
    from openai import OpenAI

    if model not in vertex_maas_models:
        raise ValueError(f"Model {model!r} not in vertex_maas_models")

    cfg = vertex_maas_models[model]
    project_id = _get_vertex_project_id()
    location = cfg["location"]
    token = _get_vertex_access_token()
    base_url = _build_vertex_base_url(project_id, location)
    return OpenAI(base_url=base_url, api_key=token, timeout=120.0), cfg


def _parse_choice(choice) -> dict:
    """Unify the parsing of a single choice into our standard format."""
    raw_content = choice.message.content
    tool_calls = _parse_tool_calls_from_choice(choice)
    content, tool_calls = _sanitize_gptoss_content(raw_content, tool_calls)
    result = {
        "content": content,
        "tool_calls": tool_calls,
        "logprobs": _parse_logprobs_from_choice(choice),
        "finish_reason": choice.finish_reason or "",
    }
    reasoning = getattr(choice.message, "reasoning_content", None)
    if not reasoning:
        extras = getattr(choice.message, "model_extra", None) or {}
        reasoning = extras.get("reasoning_content")
    if reasoning:
        result["reasoning_content"] = reasoning
    return result


def _extract_usage(response) -> dict:
    """Pull prompt/completion token counts from the API response."""
    if response.usage:
        return {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
        }
    return {"prompt_tokens": 0, "completion_tokens": 0}


@retry_with_backoff(max_retries=8, base_delay=5.0)
def generate_n_chat_samples(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    n: int = 1,
    temperature: float = 0.8,
    max_tokens: int | None = None,
    logprobs: bool = True,
) -> tuple[list[dict], dict]:
    """Generate N chat completions (with optional tool calling) in one API call.

    Args:
        model: Vertex MaaS model key (e.g. "qwen3-235b").
        messages: OpenAI-format message list.
        tools: OpenAI-format tool/function schemas, or None.
        n: Number of completions to generate.
        temperature: Sampling temperature.
        max_tokens: Max output tokens per completion. None = model default.

    Returns:
        Tuple of (samples, usage) where samples is a list of dicts with keys
        ``content, tool_calls, logprobs, finish_reason`` and usage is
        ``{"prompt_tokens": int, "completion_tokens": int}``.
    """
    client, cfg = _build_client(model)

    gen_defaults = vertex_maas_generation_defaults.get(model, {})
    params: dict = {
        "model": cfg["model_id"],
        "messages": messages,
        "temperature": gen_defaults.get("temperature", temperature),
        "logprobs": logprobs,
    }
    if n > 1:
        params["n"] = n
    if "top_p" in gen_defaults:
        params["top_p"] = gen_defaults["top_p"]
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"
    # region agent log
    _debug_log(
        "H3",
        "vertex_chat_sampler.py:215",
        "vertex_batch_request_start",
        {
            "model": model,
            "model_id": cfg["model_id"],
            "message_count": len(messages),
            "roles": [m.get("role") for m in messages],
            "total_message_chars": sum(len(m.get("content") or "") for m in messages if isinstance(m.get("content"), str)),
            "tools_count": 0 if not tools else len(tools),
            "tool_names": [] if not tools else [t.get("function", {}).get("name") for t in tools[:6]],
            "tool_schema_chars": 0 if not tools else sum(len(json.dumps(t, default=str)) for t in tools),
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )
    # endregion

    _MAX_MALFORMED_RETRIES = 3
    _last_malformed_exc = None

    for _mal_attempt in range(_MAX_MALFORMED_RETRIES):
        try:
            t0 = time.time()
            response = client.chat.completions.create(**params)
        except Exception as e:
            # region agent log
            _debug_log(
                "H3",
                "vertex_chat_sampler.py:235",
                "vertex_batch_request_exception",
                {
                    "model": model,
                    "n": n,
                    "error_type": type(e).__name__,
                    "error_preview": str(e)[:300],
                },
            )
            # endregion
            err_str = str(e).lower()
            if "n" in err_str and ("not supported" in err_str or "invalid" in err_str):
                logger.warning("n>1 not supported, falling back to sequential calls")
                # region agent log
                _debug_log(
                    "H3",
                    "vertex_chat_sampler.py:249",
                    "vertex_batch_request_fallback_sequential",
                    {"model": model, "n": n},
                )
                # endregion
                return _generate_n_sequential(
                    client, cfg, n, messages, tools, temperature, max_tokens,
                    top_p=params.get("top_p"), logprobs=logprobs,
                )
            raise

        # Guard against malformed Vertex responses (None choices, raw string, etc.)
        if not hasattr(response, "choices") or response.choices is None:
            _last_malformed_exc = RuntimeError(
                f"Vertex API returned malformed response (type={type(response).__name__}, "
                f"has_choices={hasattr(response, 'choices')}, "
                f"choices_value={repr(getattr(response, 'choices', 'MISSING'))[:200]})"
            )
            logger.warning(
                f"Malformed Vertex response (attempt {_mal_attempt + 1}/{_MAX_MALFORMED_RETRIES}): "
                f"type={type(response).__name__}, choices={getattr(response, 'choices', 'MISSING')}"
            )
            if _mal_attempt < _MAX_MALFORMED_RETRIES - 1:
                time.sleep(2 ** _mal_attempt)
                client, cfg = _build_client(model)
                continue
            logger.warning("All batch attempts returned malformed responses, falling back to streaming")
            return _generate_n_sequential(
                client, cfg, n, messages, tools, temperature, max_tokens,
                top_p=params.get("top_p"), logprobs=logprobs,
            )

        break  # valid response

    # region agent log
    _debug_log(
        "H3",
        "vertex_chat_sampler.py:260",
        "vertex_batch_request_returned",
        {
            "model": model,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "choice_count": len(response.choices),
            "usage_prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if getattr(response, "usage", None) else 0,
            "usage_completion_tokens": getattr(response.usage, "completion_tokens", 0) if getattr(response, "usage", None) else 0,
        },
    )
    # endregion

    results = [_parse_choice(c) for c in response.choices]
    usage = _extract_usage(response)
    # region agent log
    _debug_log(
        "H4",
        "vertex_chat_sampler.py:273",
        "vertex_batch_parsed_first_choice",
        {
            "model": model,
            "result_count": len(results),
            "first_content_present": bool(results and results[0].get("content")),
            "first_tool_call_count": 0 if not results or not results[0].get("tool_calls") else len(results[0]["tool_calls"]),
            "first_logprobs_len": 0 if not results else len(results[0].get("logprobs") or []),
            "first_finish_reason": "" if not results else results[0].get("finish_reason", ""),
            "first_logprob_tokens": [] if not results else [lp["token"] for lp in (results[0].get("logprobs") or [])[:8]],
        },
    )
    # endregion

    if len(results) < n:
        logger.warning(
            f"API returned {len(results)}/{n} choices, generating rest sequentially"
        )
        remaining, extra_usage = _generate_n_sequential(
            client, cfg, n - len(results), messages, tools, temperature, max_tokens,
            top_p=params.get("top_p"), logprobs=logprobs,
        )
        results.extend(remaining)
        usage["prompt_tokens"] += extra_usage["prompt_tokens"]
        usage["completion_tokens"] += extra_usage["completion_tokens"]

    return results[:n], usage


def _generate_one_streaming(
    client,
    cfg: dict,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None = None,
    logprobs: bool = True,
) -> tuple[dict, dict]:
    """Generate a single completion via streaming to capture full logprobs."""
    params: dict = {
        "model": cfg["model_id"],
        "messages": messages,
        "temperature": temperature,
        "logprobs": logprobs,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if top_p is not None:
        params["top_p"] = top_p
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"
    _debug_log(
        "H3",
        "vertex_chat_sampler.py:319",
        "vertex_stream_request_start",
        {
            "model_id": cfg["model_id"],
            "message_count": len(messages),
            "tools_count": 0 if not tools else len(tools),
            "max_tokens": max_tokens,
        },
    )
    # endregion

    t0 = time.time()
    stream = client.chat.completions.create(**params)

    content_parts: list[str] = []
    logprobs_list: list[dict] = []
    tool_call_parts: dict[int, dict] = {}
    finish_reason = ""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for chunk in stream:
        if chunk.usage:
            usage["prompt_tokens"] = chunk.usage.prompt_tokens or 0
            usage["completion_tokens"] = chunk.usage.completion_tokens or 0

        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta
        if choice.finish_reason:
            finish_reason = choice.finish_reason

        if delta.content:
            content_parts.append(delta.content)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_call_parts:
                    tool_call_parts[idx] = {
                        "id": tc_delta.id or "",
                        "name": "",
                        "arguments": "",
                    }
                if tc_delta.id:
                    tool_call_parts[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tool_call_parts[idx]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tool_call_parts[idx]["arguments"] += tc_delta.function.arguments

        if choice.logprobs and choice.logprobs.content:
            for tlp in choice.logprobs.content:
                top = []
                for alt in getattr(tlp, "top_logprobs", []) or []:
                    top.append({"token": alt.token, "logprob": alt.logprob})
                logprobs_list.append({
                    "token": tlp.token,
                    "logprob": tlp.logprob,
                    "top_logprobs": top,
                })

    content = "".join(content_parts) or None
    tool_calls = None
    if tool_call_parts:
        tool_calls = []
        for idx in sorted(tool_call_parts):
            tc = tool_call_parts[idx]
            args = tc["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": args,
            })

    result = {
        "content": content,
        "tool_calls": tool_calls,
        "logprobs": logprobs_list,
        "finish_reason": finish_reason,
    }
    # region agent log
    _debug_log(
        "H4",
        "vertex_chat_sampler.py:397",
        "vertex_stream_request_completed",
        {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "content_present": bool(content),
            "tool_call_count": 0 if not tool_calls else len(tool_calls),
            "logprobs_len": len(logprobs_list),
            "finish_reason": finish_reason,
            "first_logprob_tokens": [lp["token"] for lp in logprobs_list[:8]],
        },
    )
    # endregion
    return result, usage


def _generate_n_sequential(
    client,
    cfg: dict,
    n: int,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int | None,
    top_p: float | None = None,
    logprobs: bool = True,
) -> tuple[list[dict], dict]:
    """Generate N completions one at a time via streaming for full logprobs."""
    results = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for _ in range(n):
        result, u = _generate_one_streaming(
            client, cfg, messages, tools, temperature, max_tokens,
            top_p=top_p, logprobs=logprobs,
        )
        results.append(result)
        total_usage["prompt_tokens"] += u["prompt_tokens"]
        total_usage["completion_tokens"] += u["completion_tokens"]
    return results, total_usage
