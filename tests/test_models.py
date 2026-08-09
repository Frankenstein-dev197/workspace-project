"""Tests for the model providers."""

import pytest

from daemon_engine.models.base import BaseLLM, LLMConfig, get_default_llm
from daemon_engine.models.providers import MockProvider, OpenAIProvider, AnthropicProvider, LocalProvider


class TestMockProvider:
    def test_chat_returns_string(self):
        provider = MockProvider()
        result = provider.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_decomposition(self):
        provider = MockProvider()
        result = provider.chat([{"role": "user", "content": "Please decompose this task"}])
        assert "-" in result  # Returns subtask list format

    def test_chat_step_by_step(self):
        provider = MockProvider()
        result = provider.chat([{"role": "user", "content": "Think step by step about X"}])
        assert "1." in result or "1 " in result

    def test_embed(self):
        provider = MockProvider()
        embedding = provider.embed("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 32
        assert all(0.0 <= v <= 1.0 for v in embedding)

    def test_model_name(self):
        provider = MockProvider()
        assert provider.model_name == "mock-1"

    def test_stream(self):
        provider = MockProvider()
        chunks = list(provider.stream([{"role": "user", "content": "Hi"}]))
        assert len(chunks) == 1


class TestLLMConfig:
    def test_default_config(self):
        config = LLMConfig()
        assert config.provider == "mock"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7

    def test_custom_config(self):
        config = LLMConfig(provider="openai", model="gpt-4-turbo", temperature=0.5)
        assert config.provider == "openai"
        assert config.model == "gpt-4-turbo"
        assert config.temperature == 0.5


class TestGetDefaultLLM:
    def test_returns_mock_without_keys(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("DAEMON_LLM_PROVIDER", "")
        llm = get_default_llm()
        assert isinstance(llm, MockProvider)


class TestProviderClasses:
    def test_openai_provider_init(self):
        provider = OpenAIProvider()
        assert isinstance(provider, BaseLLM)
        assert provider.config.provider == "openai"

    def test_anthropic_provider_init(self):
        provider = AnthropicProvider()
        assert isinstance(provider, BaseLLM)
        assert provider.config.provider == "anthropic"

    def test_local_provider_init(self):
        provider = LocalProvider()
        assert isinstance(provider, BaseLLM)
        assert provider.config.provider == "local"
