"""Unified Memory: aggregates all memory subsystems into one interface.

Provides a single entry point for agents to store and retrieve information
across code, knowledge, and long-term memory layers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from daemon_engine.memory.code_memory import CodeMemory
from daemon_engine.memory.knowledge_memory import KnowledgeMemory
from daemon_engine.memory.long_term_memory import LongTermMemory, MemoryType

logger = logging.getLogger(__name__)


class UnifiedMemory:
    """Unified interface over code, knowledge, and long-term memory."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self.code_memory = CodeMemory(storage_path=self.storage_path)
        self.knowledge_memory = KnowledgeMemory(storage_path=self.storage_path)
        self.long_term_memory = LongTermMemory(storage_path=self.storage_path)

    def store(
        self,
        content: str,
        memory_type: str = "episodic",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        metadata = metadata or {}
        if memory_type == "code":
            artifact = self.code_memory.store(content=content, metadata=metadata, **kwargs)
            return artifact.id
        elif memory_type == "knowledge":
            node = self.knowledge_memory.add_knowledge(content=content, metadata=metadata, **kwargs)
            return node.id
        else:
            mt = MemoryType(memory_type) if memory_type in [m.value for m in MemoryType] else MemoryType.EPISODIC
            record = self.long_term_memory.store(
                content=content, memory_type=mt, metadata=metadata, **kwargs
            )
            return record.id

    def recall(self, query: str, limit: int = 5) -> str:
        parts: list[str] = []
        code_result = self.code_memory.recall(query, limit=limit)
        if code_result:
            parts.append(f"== Code Memory ==\n{code_result}")
        knowledge_result = self.knowledge_memory.recall(query, limit=limit)
        if knowledge_result:
            parts.append(f"== Knowledge Memory ==\n{knowledge_result}")
        long_term_result = self.long_term_memory.recall(query, limit=limit)
        if long_term_result:
            parts.append(f"== Long-Term Memory ==\n{long_term_result}")
        return "\n\n".join(parts) if parts else ""

    def search_all(self, query: str, limit: int = 5) -> dict[str, list[Any]]:
        return {
            "code": self.code_memory.search(query, limit=limit),
            "knowledge": self.knowledge_memory.search(query, limit=limit),
            "long_term": self.long_term_memory.search(query, limit=limit),
        }

    def save_all(self) -> None:
        self.code_memory.save()
        self.knowledge_memory.save()
        self.long_term_memory.save()

    def consolidate(self) -> dict[str, int]:
        pruned = self.long_term_memory.consolidate()
        return {"pruned_long_term": pruned}

    def stats(self) -> dict[str, Any]:
        return {
            "code_memory": self.code_memory.stats(),
            "knowledge_memory": self.knowledge_memory.stats(),
            "long_term_memory": self.long_term_memory.stats(),
        }

    def clear(self) -> None:
        self.code_memory.clear()
        self.knowledge_memory.clear()
        self.long_term_memory.clear()
