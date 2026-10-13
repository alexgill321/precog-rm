"""N-sample chat completion via any OpenAI-compatible endpoint (vLLM, Anyscale, etc.).

Drop-in replacement for vertex_chat_sampler.generate_n_chat_samples() when
the model is served behind a standard OpenAI-compatible API rather than
Vertex MaaS.

For models without native tool-calling support (e.g. OLMo), tools are
injected into the system prompt and tool-call JSON is parsed from the
model's text response.

Environment variables:
    OPENAI_COMPAT_BASE_URL  – e.g. https://<service>.anyscaleuserdata.com/v1
    OPENAI_COMPAT_API_KEY   – bearer token for the endpoint
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from functools import wraps

logger = logging.getLogger(__name__)

_TDP_DEBUG = os.environ.get("TDP_DEBUG", "").lower() in ("1", "true", "yes")


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    if not _TDP_DEBUG:
        return
    sys.stderr.write(f"[DEBUG] {message} | {json.dumps(data, default=str)}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Retry decorator (self-contained so we don't depend on TRAJECT-Bench)
# ---------------------------------------------------------------------------

def retry_with_backoff(max_retries: int = 8, base_delay: float = 5.0):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"{fn.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def _build_client(
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
):
    """Create an OpenAI client pointed at a generic OpenAI-compatible endpoint."""
    from openai import OpenAI

    base_url = api_base or os.environ.get("OPENAI_COMPAT_BASE_URL")
    key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "no-key")

    if not base_url:
        raise ValueError(
            "No API base URL. Set OPENAI_COMPAT_BASE_URL or pass --api-base."
        )

    return OpenAI(base_url=base_url, api_key=key, timeout=120.0)


def _is_reasoning_model(model: str) -> bool:
    """gpt-5 family models are reasoning models with different param support."""
    return "gpt-5" in (model or "").lower()


def _apply_reasoning_overrides(params: dict, model: str) -> dict:
    """For gpt-5 reasoning models: drop params they reject (custom ``temperature``,
    ``logprobs``) and request minimal thinking via the OpenAI chat-completions
    top-level ``reasoning_effort`` param (also accepted by OpenRouter)."""
    if not _is_reasoning_model(model):
        return params
    params.pop("temperature", None)
    params.pop("logprobs", None)
    params.setdefault("reasoning_effort", "minimal")
    return params


# ---------------------------------------------------------------------------
# Response parsing (shared with vertex_chat_sampler)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prompt-based tool calling (for models without native tool support)
# ---------------------------------------------------------------------------

_TOOL_PROMPT_HEADER = """You have access to the following tools. When you need to call a tool, respond with ONLY a JSON object in this exact format (no other text):
{"name": "<tool_name>", "arguments": {<arguments>}}

If you want to send a message to the user instead, just respond with plain text (no JSON wrapper).

Available tools:"""


def _render_tool_schema(tool: dict) -> str:
    """Render one OpenAI function-calling tool schema as readable text."""
    fn = tool.get("function", tool)
    name = fn["name"]
    desc = fn.get("description", "")
    params = fn.get("parameters", {})
    props = params.get("properties", {})
    required = set(params.get("required", []))

    parts = [f"- {name}: {desc}"]
    if props:
        param_strs = []
        for pname, pspec in props.items():
            ptype = pspec.get("type", "any")
            pdesc = pspec.get("description", "")
            req = " (required)" if pname in required else ""
            param_strs.append(f"    {pname} ({ptype}{req}): {pdesc}")
        parts.append("  Parameters:")
        parts.extend(param_strs)
    return "\n".join(parts)


def _build_tool_prompt(tools: list[dict]) -> str:
    """Build a tool-description block to inject into the system message."""
    rendered = "\n\n".join(_render_tool_schema(t) for t in tools)
    return f"{_TOOL_PROMPT_HEADER}\n\n{rendered}"


def _inject_tools_into_messages(
    messages: list[dict], tools: list[dict]
) -> list[dict]:
    """Return a copy of *messages* with tool definitions in the system prompt.

    Also rewrites any ``tool`` role messages to ``user`` role so that models
    without native tool support can understand the conversation history.
    """
    tool_prompt = _build_tool_prompt(tools)
    result = []
    for msg in messages:
        m = dict(msg)
        if m["role"] == "system":
            m["content"] = (m.get("content") or "") + "\n\n" + tool_prompt
            result.append(m)
        elif m["role"] == "assistant" and m.get("tool_calls"):
            tcs = m["tool_calls"]
            tc = tcs[0]
            fn = tc.get("function", tc)
            name = fn.get("name", tc.get("name", ""))
            args = fn.get("arguments", tc.get("arguments", "{}"))
            if isinstance(args, dict):
                args = json.dumps(args)
            m = {
                "role": "assistant",
                "content": json.dumps({"name": name, "arguments": json.loads(args) if isinstance(args, str) else args}),
            }
            result.append(m)
        elif m["role"] == "tool":
            result.append({
                "role": "user",
                "content": f"[Tool result for {m.get('tool_call_id', 'unknown')}]: {m.get('content', '')}",
            })
        else:
            result.append(m)
    return result


_TOOL_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{', re.DOTALL
)


def _try_parse_tool_call_from_text(text: str) -> list[dict] | None:
    """Attempt to extract a tool call from freeform model output.

    Returns a list with one tool-call dict, or None if text is not a tool call.
    """
    if not text:
        return None
    stripped = text.strip()

    if not _TOOL_CALL_RE.search(stripped):
        return None

    # Find the JSON object boundaries
    start = stripped.find("{")
    if start == -1:
        return None

    depth = 0
    end = -1
    for i in range(start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    try:
        obj = json.loads(stripped[start:end])
    except json.JSONDecodeError:
        return None

    if "name" in obj and "arguments" in obj:
        args = obj["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        return [{
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "name": obj["name"],
            "arguments": args,
        }]
    return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_logprobs_from_choice(choice) -> list[dict]:
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


def _parse_choice(choice, prompt_tools: bool = False) -> dict:
    """Parse a single API choice into our standard dict format.

    When *prompt_tools* is True, also attempt to extract a tool call from
    the model's text output (for models without native tool support).
    """
    content = choice.message.content
    tool_calls = _parse_tool_calls_from_choice(choice)

    if prompt_tools and not tool_calls and content:
        tool_calls = _try_parse_tool_call_from_text(content)
        if tool_calls:
            content = None

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
    if response.usage:
        return {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
        }
    return {"prompt_tokens": 0, "completion_tokens": 0}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@retry_with_backoff(max_retries=8, base_delay=5.0)
def generate_n_chat_samples(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    n: int = 1,
    temperature: float = 0.8,
    max_tokens: int | None = None,
    logprobs: bool = True,
    api_base: str | None = None,
    api_key: str | None = None,
) -> tuple[list[dict], dict]:
    """Generate N chat completions via an OpenAI-compatible endpoint.

    Same return signature as vertex_chat_sampler.generate_n_chat_samples:
    (samples, usage) where samples is a list of dicts with keys
    ``content, tool_calls, logprobs, finish_reason``.

    Tools are injected into the system prompt (prompt-based tool calling)
    rather than passed as structured ``tools``/``tool_choice`` parameters,
    since the target model may not support native function calling.
    """
    client = _build_client(model, api_base=api_base, api_key=api_key)

    prompt_tools = bool(tools)
    api_messages = (
        _inject_tools_into_messages(messages, tools) if tools else list(messages)
    )

    params: dict = {
        "model": model,
        "messages": api_messages,
        "temperature": temperature,
        "logprobs": logprobs,
    }
    if n > 1:
        params["n"] = n
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    params = _apply_reasoning_overrides(params, model)

    _debug_log(
        "H3",
        "openai_chat_sampler.py",
        "openai_compat_batch_request_start",
        {
            "model": model,
            "api_base": api_base or os.environ.get("OPENAI_COMPAT_BASE_URL", ""),
            "message_count": len(api_messages),
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools_count": 0 if not tools else len(tools),
            "prompt_tools": prompt_tools,
        },
    )

    try:
        t0 = time.time()
        response = client.chat.completions.create(**params)
    except Exception as e:
        _debug_log(
            "H3",
            "openai_chat_sampler.py",
            "openai_compat_batch_request_exception",
            {
                "model": model,
                "error_type": type(e).__name__,
                "error_preview": str(e)[:300],
            },
        )
        err_str = str(e).lower()
        if "n" in err_str and ("not supported" in err_str or "invalid" in err_str):
            logger.warning("n>1 not supported, falling back to sequential calls")
            return _generate_n_sequential(
                client, model, n, api_messages, None, temperature, max_tokens,
                logprobs=logprobs, prompt_tools=prompt_tools,
            )
        raise

    if not hasattr(response, "choices") or response.choices is None:
        logger.warning("Malformed response, falling back to sequential calls")
        return _generate_n_sequential(
            client, model, n, api_messages, None, temperature, max_tokens,
            logprobs=logprobs, prompt_tools=prompt_tools,
        )

    _debug_log(
        "H3",
        "openai_chat_sampler.py",
        "openai_compat_batch_request_returned",
        {
            "model": model,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "choice_count": len(response.choices),
        },
    )

    results = [_parse_choice(c, prompt_tools=prompt_tools) for c in response.choices]
    usage = _extract_usage(response)

    if len(results) < n:
        logger.warning(
            f"API returned {len(results)}/{n} choices, generating rest sequentially"
        )
        remaining, extra_usage = _generate_n_sequential(
            client, model, n - len(results), api_messages, None, temperature,
            max_tokens, logprobs=logprobs, prompt_tools=prompt_tools,
        )
        results.extend(remaining)
        usage["prompt_tokens"] += extra_usage["prompt_tokens"]
        usage["completion_tokens"] += extra_usage["completion_tokens"]

    return results[:n], usage


# ---------------------------------------------------------------------------
# Streaming fallback
# ---------------------------------------------------------------------------

def _generate_one_streaming(
    client,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int | None,
    logprobs: bool = True,
    prompt_tools: bool = False,
) -> tuple[dict, dict]:
    params: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "logprobs": logprobs,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"
    params = _apply_reasoning_overrides(params, model)

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

    if prompt_tools and not tool_calls and content:
        parsed = _try_parse_tool_call_from_text(content)
        if parsed:
            tool_calls = parsed
            content = None

    result = {
        "content": content,
        "tool_calls": tool_calls,
        "logprobs": logprobs_list,
        "finish_reason": finish_reason,
    }
    _debug_log(
        "H4",
        "openai_chat_sampler.py",
        "openai_compat_stream_completed",
        {
            "elapsed_ms": int((time.time() - t0) * 1000),
            "content_present": bool(content),
            "tool_call_count": 0 if not tool_calls else len(tool_calls),
            "logprobs_len": len(logprobs_list),
            "finish_reason": finish_reason,
        },
    )
    return result, usage


def _generate_n_sequential(
    client,
    model: str,
    n: int,
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int | None,
    logprobs: bool = True,
    prompt_tools: bool = False,
) -> tuple[list[dict], dict]:
    results = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for _ in range(n):
        result, u = _generate_one_streaming(
            client, model, messages, tools, temperature, max_tokens,
            logprobs=logprobs, prompt_tools=prompt_tools,
        )
        results.append(result)
        total_usage["prompt_tokens"] += u["prompt_tokens"]
        total_usage["completion_tokens"] += u["completion_tokens"]
    return results, total_usage
