"""Vertex MaaS handler subclasses for BFCL models (Qwen, DeepSeek).

Reuses upstream handler logic but swaps the OpenAI client to point at
the Vertex AI endpoint with auto-refreshing GCP access tokens.

Each API call builds a thread-local client with a fresh token to avoid
races when running with concurrent threads (in-process mode).
"""

from __future__ import annotations

import os
import re
import threading
import time

from openai import OpenAI


# ---------------------------------------------------------------------------
# Vertex MaaS auth helpers (ported from TRAJECT-Bench utils/model_providers)
# ---------------------------------------------------------------------------

_vertex_credentials_cache = None
_vertex_token_lock = threading.Lock()
_vertex_token_cache: dict = {"token": None, "expiry": 0}


def _get_vertex_access_token() -> str:
    """Obtain a cached GCP access token, refreshed 60s before expiry."""
    import google.auth
    import google.auth.transport.requests

    global _vertex_credentials_cache

    with _vertex_token_lock:
        if _vertex_token_cache["token"] and time.time() < _vertex_token_cache["expiry"]:
            return _vertex_token_cache["token"]

        if _vertex_credentials_cache is None:
            _vertex_credentials_cache, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        request = google.auth.transport.requests.Request()
        _vertex_credentials_cache.refresh(request)

        expiry = getattr(_vertex_credentials_cache, "expiry", None)
        if expiry:
            _vertex_token_cache["expiry"] = expiry.timestamp() - 60
        else:
            _vertex_token_cache["expiry"] = time.time() + 3000

        _vertex_token_cache["token"] = _vertex_credentials_cache.token
        return _vertex_token_cache["token"]


def _get_vertex_project_id() -> str:
    project_id = os.getenv("GCP_PROJECT_ID")
    if project_id:
        return project_id
    try:
        import google.auth
        _, project = google.auth.default()
        if project:
            return project
    except Exception:
        pass
    raise ValueError("Set GCP_PROJECT_ID env var or configure gcloud default project")


def _build_vertex_base_url(project_id: str, location: str) -> str:
    if location == "global":
        return (
            f"https://aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/global/endpoints/openapi"
        )
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/endpoints/openapi"
    )


# ---------------------------------------------------------------------------
# Model registry — maps our shorthand names to Vertex model IDs
# ---------------------------------------------------------------------------

VERTEX_QWEN_MODELS: dict[str, dict] = {
    "qwen3-next-80b-thinking-vertex": {
        "model_id": "qwen/qwen3-next-80b-a3b-thinking-maas",
        "display_name": "Qwen3-Next-80B-Thinking (Vertex)",
        "location": "global",
    },
    "qwen3-next-80b-vertex": {
        "model_id": "qwen/qwen3-next-80b-a3b-instruct-maas",
        "display_name": "Qwen3-Next-80B-Instruct (Vertex)",
        "location": "global",
    },
}

VERTEX_DEEPSEEK_MODELS: dict[str, dict] = {
    "deepseek-v3.2-vertex": {
        "model_id": "deepseek-ai/deepseek-v3.2-maas",
        "display_name": "DeepSeek-V3.2 (Vertex)",
        "location": "global",
    },
}


def _build_vertex_client(vertex_cfg: dict) -> OpenAI:
    project_id = _get_vertex_project_id()
    location = vertex_cfg["location"]
    token = _get_vertex_access_token()
    base_url = _build_vertex_base_url(project_id, location)
    return OpenAI(base_url=base_url, api_key=token, timeout=120.0)


# ---------------------------------------------------------------------------
# Shared init logic
# ---------------------------------------------------------------------------

def _vertex_handler_init(handler, model_registry: dict, env_keys: list[str], **init_kwargs):
    """Common init for Vertex MaaS handlers.

    Calls the grandparent OpenAICompletionsHandler.__init__ (skipping the
    provider-specific parent that hardcodes DashScope / DeepSeek endpoints),
    then stores the Vertex config for per-call client creation.
    """
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.constants.enums import ModelStyle

    registry_name = init_kwargs["registry_name"]

    saved = {}
    for key in env_keys:
        if key not in os.environ:
            os.environ[key] = "placeholder-for-vertex"
            saved[key] = True
    try:
        OpenAICompletionsHandler.__init__(handler, **init_kwargs)
    finally:
        for key in saved:
            os.environ.pop(key, None)
    handler.model_style = ModelStyle.OPENAI_COMPLETIONS

    handler._vertex_cfg = model_registry.get(registry_name)
    if handler._vertex_cfg is None:
        raise ValueError(
            f"No Vertex config for {registry_name!r}. "
            f"Known: {list(model_registry)}"
        )
    # Build an initial client so the handler is usable immediately
    # (e.g. for subprocess mode where there's no thread contention).
    handler.client = _build_vertex_client(handler._vertex_cfg)


def _invalidate_token_cache():
    """Force the next _get_vertex_access_token call to refresh."""
    with _vertex_token_lock:
        _vertex_token_cache["token"] = None
        _vertex_token_cache["expiry"] = 0


def _vertex_generate_with_backoff(handler, super_generate, **kwargs):
    """Make the API call with a fresh client, retrying on transient errors.

    Builds a new OpenAI client (with a fresh GCP token) for every attempt,
    so token expiry and thread races are handled naturally.
    """
    import time as _time
    from openai import RateLimitError, AuthenticationError

    if "model" in kwargs:
        kwargs["model"] = handler._vertex_cfg["model_id"]

    max_retries = 8
    base_delay = 5.0
    for attempt in range(max_retries):
        client = _build_vertex_client(handler._vertex_cfg)
        try:
            start = _time.time()
            response = client.chat.completions.create(**kwargs)
            return response, _time.time() - start
        except AuthenticationError:
            _invalidate_token_cache()
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), 120.0)
            _time.sleep(delay)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), 120.0)
            _time.sleep(delay)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

from bfcl_eval.model_handler.api_inference.qwen import QwenAPIHandler  # noqa: E402


class VertexQwenHandler(QwenAPIHandler):
    """QwenAPIHandler wired to Vertex MaaS with per-call token refresh."""

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        _vertex_handler_init(
            self, VERTEX_QWEN_MODELS,
            env_keys=["OPENAI_API_KEY"],
            model_name=model_name, temperature=temperature,
            registry_name=registry_name, is_fc_model=is_fc_model, **kwargs,
        )

    def generate_with_backoff(self, **kwargs):
        return _vertex_generate_with_backoff(
            self, super().generate_with_backoff, **kwargs,
        )


from bfcl_eval.model_handler.api_inference.deepseek import DeepSeekAPIHandler  # noqa: E402


class VertexDeepSeekHandler(DeepSeekAPIHandler):
    """DeepSeekAPIHandler wired to Vertex MaaS with per-call token refresh."""

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        _vertex_handler_init(
            self, VERTEX_DEEPSEEK_MODELS,
            env_keys=["DEEPSEEK_API_KEY", "OPENAI_API_KEY"],
            model_name=model_name, temperature=temperature,
            registry_name=registry_name, is_fc_model=is_fc_model, **kwargs,
        )

    def generate_with_backoff(self, **kwargs):
        return _vertex_generate_with_backoff(
            self, super().generate_with_backoff, **kwargs,
        )

    def _query_prompting(self, inference_data: dict):
        """Override DeepSeekAPIHandler._query_prompting which drops temperature
        (designed for deepseek-reasoner). V3.2 on Vertex supports temperature."""
        inference_data["inference_input_log"] = {"message": repr(inference_data["message"])}
        return self.generate_with_backoff(
            model=self.model_name,
            messages=inference_data["message"],
            temperature=self.temperature,
        )

    def decode_execute(self, result, has_tool_call_tag=False):
        """Extract tool calls from mixed text+function output.

        DeepSeek on Vertex often prepends reasoning text before the
        ``[func(...)]`` block.  The upstream parser expects the entire
        response to be valid Python, so we extract the bracket-delimited
        portion first.
        """
        if not self.is_fc_model and isinstance(result, str):
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group(0)
        return super().decode_execute(result, has_tool_call_tag=has_tool_call_tag)


# ---------------------------------------------------------------------------
# OpenRouter handler (chat-completions; used for gpt-5-nano etc.)
# ---------------------------------------------------------------------------

from bfcl_eval.model_handler.api_inference.openai_completion import (  # noqa: E402
    OpenAICompletionsHandler,
)
from bfcl_eval.constants.enums import ModelStyle  # noqa: E402

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_MODELS: dict[str, dict] = {
    "gpt-5-nano-openrouter": {
        "model_id": "openai/gpt-5-nano",
        "display_name": "GPT-5-nano (OpenRouter)",
        "reasoning_effort": "minimal",
    },
    "gpt-5-mini-openrouter": {
        "model_id": "openai/gpt-5-mini",
        "display_name": "GPT-5-mini (OpenRouter)",
        "reasoning_effort": "minimal",
    },
}


def _is_reasoning_model(model_id: str) -> bool:
    """gpt-5 family models are reasoning models with restricted params."""
    return "gpt-5" in (model_id or "").lower()


def _build_openrouter_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to the run's .env "
            "(passed via --dotenv) or export it before running."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=120.0)


class OpenRouterHandler(OpenAICompletionsHandler):
    """OpenAI chat-completions handler pointed at OpenRouter.

    Used for models (e.g. gpt-5-nano) that OpenRouter serves via the
    chat-completions API. For gpt-5 reasoning models, per-call params are
    adjusted: ``temperature``/``logprobs`` are dropped (unsupported) and a
    minimal reasoning-effort is requested.
    """

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        # The grandparent __init__ builds an OpenAI() client from OPENAI_API_KEY,
        # which we don't need — seed a placeholder so construction succeeds, then
        # swap in the OpenRouter client.
        saved = False
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "placeholder-for-openrouter"
            saved = True
        try:
            OpenAICompletionsHandler.__init__(
                self, model_name, temperature, registry_name, is_fc_model, **kwargs,
            )
        finally:
            if saved:
                os.environ.pop("OPENAI_API_KEY", None)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self._reasoning_effort = OPENROUTER_MODELS.get(registry_name, {}).get(
            "reasoning_effort"
        )
        self.client = _build_openrouter_client()

    def generate_with_backoff(self, **kwargs):
        model_id = kwargs.get("model", self.model_name)
        if _is_reasoning_model(model_id):
            kwargs.pop("temperature", None)
            kwargs.pop("logprobs", None)
            if self._reasoning_effort:
                extra = dict(kwargs.get("extra_body") or {})
                extra.setdefault("reasoning", {"effort": self._reasoning_effort})
                kwargs["extra_body"] = extra
        return super().generate_with_backoff(**kwargs)

    def decode_execute(self, result, has_tool_call_tag=False):
        """Prompt-based models may prepend prose before the ``[func(...)]`` block;
        extract the bracketed portion so the upstream parser can read it."""
        if not self.is_fc_model and isinstance(result, str):
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                result = match.group(0)
        return super().decode_execute(result, has_tool_call_tag=has_tool_call_tag)
