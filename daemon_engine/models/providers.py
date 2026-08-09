"""LLM provider implementations."""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from daemon_engine.models.base import BaseLLM, LLMConfig

logger = logging.getLogger(__name__)


class MockProvider(BaseLLM):
    """A mock LLM provider for testing and development without API keys.

    Returns deterministic responses based on input hashing, so tests are
    reproducible. Inspired by the test fixtures pattern from DeerFlow.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config or LLMConfig(provider="mock", model="mock-1"))

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        last_msg = messages[-1]["content"] if messages else ""
        seed = int(hashlib.md5(last_msg.encode()).hexdigest()[:8], 16)
        if "decompose" in last_msg.lower() or "break down" in last_msg.lower():
            return (
                "- Analyze the requirements and constraints\n"
                "- Design the solution architecture\n"
                "- Implement the core components\n"
                "- Write tests and verify functionality\n"
                "- Document and deploy"
            )
        if "step by step" in last_msg.lower():
            return (
                "1. First, I'll analyze the problem structure.\n"
                "2. Next, I'll identify key variables and constraints.\n"
                "3. Then, I'll formulate an approach.\n"
                "4. I'll execute the approach step by step.\n"
                "5. Finally, I'll verify the result and conclude.\n"
                f"Final answer: Based on analysis of '{last_msg[:50]}...', the solution is feasible."
            )
        if "select the best option" in last_msg.lower():
            return "SELECTED: opt_1\nREASONING: The first option has the highest expected utility."
        return (
            f"[MockLLM] Processed: {last_msg[:100]}... "
            f"Final answer: Task complete. Seed={seed}"
        )

    def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:32]]


class OpenAIProvider(BaseLLM):
    """OpenAI-compatible provider (works with OpenAI and compatible APIs)."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig(
            provider="openai",
            model=os.environ.get("DAEMON_LLM_MODEL", "gpt-4"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        super().__init__(cfg)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc
        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return response.choices[0].message.content or ""

    def embed(self, text: str) -> list[float]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package required") from exc
        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        response = client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding


class AnthropicProvider(BaseLLM):
    """Anthropic Claude provider."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig(
            provider="anthropic",
            model=os.environ.get("DAEMON_LLM_MODEL", "claude-sonnet-4-20250514"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        )
        super().__init__(cfg)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package required: pip install anthropic") from exc
        client = anthropic.Anthropic(api_key=self.config.api_key, base_url=self.config.base_url)
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)
        response = client.messages.create(
            model=self.config.model,
            system=system_msg,
            messages=chat_messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
        )
        return response.content[0].text if response.content else ""

    def embed(self, text: str) -> list[float]:
        logger.warning("Anthropic does not provide embeddings, using mock")
        return MockProvider().embed(text)


class LocalProvider(BaseLLM):
    """Local model provider using Transformers or Ollama.

    Integrates the Transformers library for local model inference and
    supports Ollama for local serving.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig(
            provider="local",
            model=os.environ.get("DAEMON_LOCAL_MODEL", "ollama/llama3"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        super().__init__(cfg)

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        base_url = self.config.base_url or "http://localhost:11434"
        try:
            import urllib.request
            import json

            data = json.dumps({
                "model": self.config.model.split("/")[-1],
                "messages": messages,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                result = json.loads(resp.read())
            return result.get("message", {}).get("content", "")
        except Exception as exc:
            logger.error("Local provider error: %s", exc)
            return f"[LocalProvider error: {exc}]"

    def embed(self, text: str) -> list[float]:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            tokenizer = AutoTokenizer.from_pretrained(self.config.model)
            model = AutoModel.from_pretrained(self.config.model)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        except Exception as exc:
            logger.warning("Transformers embedding failed, using mock: %s", exc)
            return MockProvider().embed(text)
