"""Vertex AI Model-as-a-Service access for the two open-weight policies.

Extracted from the (locally modified) TRAJECT-Bench `utils/model_providers.py`
the original runs imported; only what the tau2 sampler and runner use.

Auth is Google Application Default Credentials (`gcloud auth application-default
login`); the project comes from `GCP_PROJECT_ID` or the ADC default project.
"""
import logging
import os
import threading
import time
from functools import wraps

vertex_maas_models = {
    "deepseek-v3.2": {
        "model_id": "deepseek-ai/deepseek-v3.2-maas",
        "display_name": "DeepSeek-V3.2",
        "location": "global",
    },
    "qwen3-next-80b-thinking": {
        "model_id": "qwen/qwen3-next-80b-a3b-thinking-maas",
        "display_name": "Qwen3-Next-80B-Thinking",
        "location": "global",
    },
}
vertex_maas_generation_defaults = {
    "deepseek-v3.2": {"temperature": 1.0},
    "qwen3-next-80b-thinking": {"temperature": 0.6, "top_p": 0.95},
}

_vertex_credentials_cache = None
_vertex_token_lock = threading.Lock()
_vertex_token_cache = {"token": None, "expiry": 0}


def _get_vertex_access_token() -> str:
    """Cached ADC access token, refreshed 60 s before expiry (thread-safe)."""
    import google.auth
    import google.auth.transport.requests
    global _vertex_credentials_cache
    with _vertex_token_lock:
        if _vertex_token_cache["token"] and time.time() < _vertex_token_cache["expiry"]:
            return _vertex_token_cache["token"]
        if _vertex_credentials_cache is None:
            _vertex_credentials_cache, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
        _vertex_credentials_cache.refresh(google.auth.transport.requests.Request())
        expiry = getattr(_vertex_credentials_cache, "expiry", None)
        _vertex_token_cache["expiry"] = expiry.timestamp() - 60 if expiry else time.time() + 3000
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
    raise ValueError("Set GCP_PROJECT_ID or configure a gcloud default project")


def _build_vertex_base_url(project_id: str, location: str) -> str:
    if location == "global":
        return f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/endpoints/openapi"
    return (f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/{location}/endpoints/openapi")


def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """Retry on rate limits, transient API errors and auth expiry (forces a
    token refresh on 401 before retrying)."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from openai import APIStatusError, AuthenticationError, RateLimitError
            except ImportError:  # pragma: no cover
                APIStatusError = AuthenticationError = RateLimitError = ()
            last = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last = e
                    msg = str(e)
                    is_auth = isinstance(e, AuthenticationError) or "401" in msg or "UNAUTHENTICATED" in msg
                    is_rate = isinstance(e, RateLimitError) or "429" in msg
                    is_server = isinstance(e, APIStatusError) and getattr(e, "status_code", 0) >= 500
                    is_conn = any(k in type(e).__name__ for k in ("Connection", "Timeout"))
                    if not (is_auth or is_rate or is_server or is_conn):
                        raise
                    if attempt == max_retries:
                        break
                    if is_auth:
                        _vertex_token_cache["token"] = None
                        _vertex_token_cache["expiry"] = 0
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logging.warning(f"{type(e).__name__}: retrying in {delay:.0f}s "
                                    f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
            raise last
        return wrapper
    return decorator


class CostTracker:
    """Accumulates token usage across API calls and computes a running $ cost."""

    PRICING = {   # $ per 1M tokens (input, output) as billed during the runs
        "deepseek-v3.2": {"input": 0.30, "output": 1.20},
        "qwen3-next-80b-thinking": {"input": 0.15, "output": 1.20},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5-mini-2025-08-07": {"input": 0.25, "output": 2.00},
    }

    def __init__(self):
        self.totals = {}
        self._lock = threading.Lock()

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        with self._lock:
            t = self.totals.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
            t["input_tokens"] += prompt_tokens
            t["output_tokens"] += completion_tokens

    def _cost_for(self, model: str) -> float:
        t = self.totals.get(model, {})
        p = self.PRICING.get(model, {"input": 0.0, "output": 0.0})
        return t.get("input_tokens", 0) / 1e6 * p["input"] + t.get("output_tokens", 0) / 1e6 * p["output"]

    @property
    def total_cost(self) -> float:
        return sum(self._cost_for(m) for m in self.totals)

    def summary(self) -> str:
        lines = ["=== Cost Summary ==="]
        for model, t in sorted(self.totals.items()):
            lines.append(f"  {model}: {t['input_tokens']:,} input / {t['output_tokens']:,} output "
                         f"= ${self._cost_for(model):.4f}")
        lines.append(f"  Total: ${self.total_cost:.4f}")
        return "\n".join(lines)
