"""Base LLM interface and configuration."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str = "mock"
    model: str = "gpt-4"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60
    extra: dict = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send a chat completion request and return the response text."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""

    def stream(self, messages: list[dict[str, str]], **kwargs):
        """Stream a chat completion. Default: yield the full response."""
        yield self.chat(messages, **kwargs)

    @property
    def model_name(self) -> str:
        return self.config.model


def get_default_llm() -> BaseLLM:
    """Return a default LLM based on available environment variables."""
    from daemon_engine.models.providers import MockProvider, OpenAIProvider, AnthropicProvider

    provider = os.environ.get("DAEMON_LLM_PROVIDER", "").lower()
    if provider == "openai" or os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider()
    if provider == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    logger.info("No LLM API key found, using MockProvider")
    return MockProvider()
