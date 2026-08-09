"""Model integration: LLM providers and reasoning models.

Provides a unified interface to multiple LLM providers, inspired by
Transformers (model abstractions), DeepSeek-Reasonix (bounded LLM), and
LangChain (chat model interface). Supports OpenAI, Anthropic, local models,
and a mock provider for testing.
"""

from daemon_engine.models.base import BaseLLM, get_default_llm, LLMConfig
from daemon_engine.models.providers import (
    OpenAIProvider,
    AnthropicProvider,
    MockProvider,
    LocalProvider,
)

__all__ = [
    "BaseLLM",
    "LLMConfig",
    "get_default_llm",
    "OpenAIProvider",
    "AnthropicProvider",
    "MockProvider",
    "LocalProvider",
]
