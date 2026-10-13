"""Drop-in replacement for the ``bfcl`` CLI with Vertex MaaS models injected.

Usage (from the flow_eval root, inside the bfcl venv):
    python -m src.bfcl_wrap.bfcl_patched generate --model qwen3-next-80b-thinking-vertex ...
    python -m src.bfcl_wrap.bfcl_patched generate --model deepseek-v3.2-vertex ...

This module monkey-patches ``MODEL_CONFIG_MAPPING`` before delegating to the
upstream ``bfcl`` CLI, so all standard bfcl flags and commands work unchanged.
"""

from __future__ import annotations


def _patch_model_config() -> None:
    """Register Vertex MaaS models into BFCL's MODEL_CONFIG_MAPPING."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from src.bfcl_wrap.vertex_bfcl_handler import (
        VERTEX_QWEN_MODELS,
        VertexQwenHandler,
        VERTEX_DEEPSEEK_MODELS,
        VertexDeepSeekHandler,
        OPENROUTER_MODELS,
        OpenRouterHandler,
    )

    for registry_name, vcfg in VERTEX_QWEN_MODELS.items():
        if registry_name in MODEL_CONFIG_MAPPING:
            continue
        MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
            model_name=vcfg["model_id"],
            display_name=vcfg["display_name"],
            url="https://cloud.google.com/vertex-ai/docs/model-as-a-service",
            org="Qwen (via Vertex MaaS)",
            license="apache-2.0",
            model_handler=VertexQwenHandler,
            input_price=None,
            output_price=None,
            is_fc_model=True,
            underscore_to_dot=True,
        )

    for registry_name, vcfg in VERTEX_DEEPSEEK_MODELS.items():
        if registry_name in MODEL_CONFIG_MAPPING:
            continue
        MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
            model_name=vcfg["model_id"],
            display_name=vcfg["display_name"],
            url="https://cloud.google.com/vertex-ai/docs/model-as-a-service",
            org="DeepSeek (via Vertex MaaS)",
            license="mit",
            model_handler=VertexDeepSeekHandler,
            input_price=None,
            output_price=None,
            is_fc_model=False,
            underscore_to_dot=False,
        )

    for registry_name, ocfg in OPENROUTER_MODELS.items():
        if registry_name in MODEL_CONFIG_MAPPING:
            continue
        MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
            model_name=ocfg["model_id"],
            display_name=ocfg["display_name"],
            url="https://openrouter.ai",
            org="OpenAI (via OpenRouter)",
            license="proprietary",
            model_handler=OpenRouterHandler,
            input_price=None,
            output_price=None,
            # Native function calling: gpt-5-nano emits structured tool_calls,
            # not the Python-style [func(...)] text the prompting decoder expects.
            is_fc_model=True,
            underscore_to_dot=True,
        )


_patch_model_config()

if __name__ == "__main__":
    from bfcl_eval.__main__ import cli

    cli()
