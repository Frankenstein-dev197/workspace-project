"""Long-Term Memory: persistent episodic and experiential memory.

Inspired by DeepSeek-Reasonix's hierarchical memory (REASONIX.md / MEMORY.md)
and learn-claude-code's session memory. Stores experiences, outcomes, and
learned patterns that persist across sessions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    EPISODIC = "episodic"  # specific events/experiences
    SEMANTIC = "semantic"  # general knowledge/facts
    PROCEDURAL = "procedural"  # how-to / skills
    FEEDBACK = "feedback"  # user corrections/preferences
    PATTERN = "pattern"  # recurring patterns learned


@dataclass
class MemoryRecord:
    id: str = ""
    content: str = ""
    memory_type: MemoryType = MemoryType.EPISODIC
    scope: str = "project"  # project or global
    embedding: list[float] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 1.0
    source: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_recalled: float = field(default_factory=time.time)
    recall_count: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.content.encode()).hexdigest()[:16]


class LongTermMemory:
    """Persistent memory with importance scoring and decay-based recall."""

    DECAY_FACTOR = 0.995
    MIN_IMPORTANCE = 0.01

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._records: dict[str, MemoryRecord] = {}
        if self.storage_path:
            self._load()

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        scope: str = "project",
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            content=content,
            memory_type=memory_type,
            scope=scope,
            importance=importance,
            confidence=confidence,
            source=source,
            tags=tags or [],
            metadata=metadata or {},
            embedding=self._embed(content),
        )
        self._records[record.id] = record
        logger.debug("Stored %s memory: %s", memory_type.value, record.id)
        return record

    def recall(self, query: str, limit: int = 5, scope: str | None = None) -> str:
        results = self.search(query, limit=limit, scope=scope)
        if not results:
            return ""
        parts: list[str] = []
        for record in results:
            parts.append(f"[{record.memory_type.value}] {record.content[:300]}")
        return "\n---\n".join(parts)

    def search(
        self, query: str, limit: int = 5, scope: str | None = None, memory_type: MemoryType | None = None
    ) -> list[MemoryRecord]:
        query_emb = self._embed(query)
        scored: list[tuple[float, MemoryRecord]] = []
        for record in self._records.values():
            if scope and record.scope != scope:
                continue
            if memory_type and record.memory_type != memory_type:
                continue
            sim = self._cosine_sim(query_emb, record.embedding)
            recency = self._recency_score(record)
            score = sim * 0.4 + record.importance * 0.4 + recency * 0.2
            scored.append((score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[MemoryRecord] = []
        for _, record in scored[:limit]:
            record.last_recalled = time.time()
            record.recall_count += 1
            results.append(record)
        return results

    def consolidate(self) -> int:
        """Apply importance decay and prune low-importance memories."""
        pruned = 0
        to_remove: list[str] = []
        for record_id, record in self._records.items():
            record.importance *= self.DECAY_FACTOR
            if record.importance < self.MIN_IMPORTANCE and record.recall_count == 0:
                to_remove.append(record_id)
        for rid in to_remove:
            del self._records[rid]
            pruned += 1
        logger.info("Consolidated memory: pruned %d records", pruned)
        return pruned

    def reinforce(self, record_id: str, factor: float = 1.5) -> bool:
        record = self._records.get(record_id)
        if not record:
            return False
        record.importance = min(record.importance * factor, 1.0)
        record.recall_count += 1
        return True

    def get_by_type(self, memory_type: MemoryType) -> list[MemoryRecord]:
        return [r for r in self._records.values() if r.memory_type == memory_type]

    def get_by_tag(self, tag: str) -> list[MemoryRecord]:
        return [r for r in self._records.values() if tag in r.tags]

    def _recency_score(self, record: MemoryRecord) -> float:
        age = time.time() - record.last_recalled
        return max(1.0 / (1.0 + age / 3600.0), 0.0)

    def _embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:32]]

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def stats(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "by_type": {mt.value: sum(1 for r in self._records.values() if r.memory_type == mt) for mt in MemoryType},
            "avg_importance": (
                sum(r.importance for r in self._records.values()) / len(self._records)
                if self._records else 0.0
            ),
            "total_recalls": sum(r.recall_count for r in self._records.values()),
        }

    def _load(self) -> None:
        if not self.storage_path:
            return
        path = self.storage_path / "long_term_memory.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for item in data:
                if "memory_type" in item and isinstance(item["memory_type"], str):
                    item["memory_type"] = MemoryType(item["memory_type"])
                record = MemoryRecord(**item)
                self._records[record.id] = record
            logger.info("Loaded %d long-term memories", len(self._records))
        except Exception as exc:
            logger.error("Failed to load long-term memory: %s", exc)

    def save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        path = self.storage_path / "long_term_memory.json"
        data = []
        for record in self._records.values():
            d = asdict(record)
            d["memory_type"] = record.memory_type.value
            data.append(d)
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved %d long-term memories", len(data))

    def clear(self) -> None:
        self._records.clear()
