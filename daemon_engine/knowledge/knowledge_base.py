"""Knowledge base: structured knowledge storage with semantic search.

Integrates Dify's knowledge segmentation model (chunking, indexing, retrieval)
with a lightweight semantic search engine. Stores knowledge entries with
metadata, tags, and content for agent retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KnowledgeSource(str, Enum):
    LEETCODE = "leetcode"
    DEVOPS = "devops"
    REFERENCE = "reference"
    CUSTOM = "custom"
    WEB = "web"
    CODE = "code"


@dataclass
class KnowledgeEntry:
    id: str
    title: str
    content: str
    source: KnowledgeSource = KnowledgeSource.CUSTOM
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    chunks: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content[:500],
            "source": self.source.value,
            "category": self.category,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "chunk_count": len(self.chunks),
        }


class KnowledgeBase:
    """Structured knowledge storage with chunking and semantic search.

    Inspired by Dify's knowledge base segmentation pipeline:
    1. Split content into chunks (by paragraphs or sentences)
    2. Generate lightweight embeddings (TF-IDF style)
    3. Index for cosine similarity search
    """

    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50
    STOP_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "up", "down", "out",
        "off", "over", "under", "again", "further", "then", "once", "and",
        "but", "or", "nor", "not", "no", "so", "than", "too", "very", "just",
    })

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        storage_path: Path | str | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage_path = Path(storage_path) if storage_path else None
        self._entries: dict[str, KnowledgeEntry] = {}
        self._chunk_index: dict[str, dict[str, float]] = {}  # entry_id -> {chunk: tfidf}
        self._idf: dict[str, float] = {}
        self._vocabulary: set[str] = set()
        self._total_chunks: int = 0

    def add_entry(
        self,
        title: str,
        content: str,
        source: KnowledgeSource = KnowledgeSource.CUSTOM,
        category: str = "general",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> KnowledgeEntry:
        entry_id = entry_id or hashlib.sha256(
            f"{title}:{content[:100]}".encode()
        ).hexdigest()[:12]
        chunks = self._chunk_content(content)
        embedding = self._compute_embedding(content)
        entry = KnowledgeEntry(
            id=entry_id,
            title=title,
            content=content,
            source=source,
            category=category,
            tags=tags or [],
            metadata=metadata or {},
            chunks=chunks,
            embedding=embedding,
        )
        self._entries[entry_id] = entry
        self._index_entry(entry)
        logger.debug("Added knowledge entry '%s' with %d chunks", title, len(chunks))
        return entry

    def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        return self._entries.get(entry_id)

    def remove_entry(self, entry_id: str) -> bool:
        entry = self._entries.pop(entry_id, None)
        if entry:
            self._chunk_index.pop(entry_id, None)
            self._rebuild_idf()
            return True
        return False

    def search(
        self,
        query: str,
        limit: int = 10,
        source: KnowledgeSource | None = None,
        category: str | None = None,
    ) -> list[tuple[float, KnowledgeEntry]]:
        query_embedding = self._compute_embedding(query)
        results: list[tuple[float, KnowledgeEntry]] = []
        for entry in self._entries.values():
            if source and entry.source != source:
                continue
            if category and entry.category != category:
                continue
            score = self._cosine_similarity(query_embedding, entry.embedding)
            query_lower = query.lower()
            if query_lower in entry.title.lower():
                score += 0.3
            for tag in entry.tags:
                if query_lower in tag.lower():
                    score += 0.2
            score += self._chunk_match_score(query, entry)
            if score > 0:
                results.append((score, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:limit]

    def get_by_tag(self, tag: str) -> list[KnowledgeEntry]:
        return [e for e in self._entries.values() if tag in e.tags]

    def get_by_category(self, category: str) -> list[KnowledgeEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def get_by_source(self, source: KnowledgeSource) -> list[KnowledgeEntry]:
        return [e for e in self._entries.values() if e.source == source]

    def all_entries(self) -> list[KnowledgeEntry]:
        return list(self._entries.values())

    def _chunk_content(self, content: str) -> list[str]:
        if len(content) <= self.chunk_size:
            return [content]
        chunks: list[str] = []
        paragraphs = content.split("\n\n")
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap = current_chunk[-self.chunk_overlap :] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap + "\n" + para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def _compute_embedding(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * 64
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        total_docs = len(self._entries) + 1
        tfidf_vec: dict[str, float] = {}
        for word, count in tf.items():
            tf_val = count / len(tokens)
            doc_freq = sum(1 for e in self._entries.values() if word in self._vocabulary)
            idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1
            tfidf_vec[word] = tf_val * idf
            self._vocabulary.add(word)
        return self._dict_to_fixed_vector(tfidf_vec)

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"\b[a-z]{2,}\b", text.lower())
        return [w for w in words if w not in self.STOP_WORDS]

    def _dict_to_fixed_vector(self, vec: dict[str, float], size: int = 64) -> list[float]:
        result = [0.0] * size
        for i, word in enumerate(sorted(vec.keys())):
            if i >= size:
                break
            result[i] = vec[word]
        norm = math.sqrt(sum(v * v for v in result))
        if norm > 0:
            result = [v / norm for v in result]
        return result

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _index_entry(self, entry: KnowledgeEntry) -> None:
        tf_map: dict[str, float] = {}
        for chunk in entry.chunks:
            tokens = self._tokenize(chunk)
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1
        self._chunk_index[entry.id] = tf_map
        self._total_chunks += len(entry.chunks)
        self._rebuild_idf()

    def _rebuild_idf(self) -> None:
        total = len(self._entries)
        if total == 0:
            self._idf = {}
            return
        doc_freq: dict[str, int] = {}
        for tf_map in self._chunk_index.values():
            for word in tf_map:
                doc_freq[word] = doc_freq.get(word, 0) + 1
        self._idf = {word: math.log((total + 1) / (df + 1)) + 1 for word, df in doc_freq.items()}

    def _chunk_match_score(self, query: str, entry: KnowledgeEntry) -> float:
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return 0.0
        tf_map = self._chunk_index.get(entry.id, {})
        if not tf_map:
            return 0.0
        score = 0.0
        for token in query_tokens:
            if token in tf_map:
                idf = self._idf.get(token, 1.0)
                score += tf_map[token] * idf
        return min(score / 100, 0.5)

    def save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [e.__dict__ for e in self._entries.values()],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        (self.storage_path / "knowledge_base.json").write_text(
            json.dumps(data, default=str, indent=2), encoding="utf-8"
        )
        logger.info("Saved %d entries to %s", len(self._entries), self.storage_path)

    def load(self) -> None:
        if not self.storage_path:
            return
        file = self.storage_path / "knowledge_base.json"
        if not file.exists():
            return
        data = json.loads(file.read_text(encoding="utf-8"))
        self._entries.clear()
        self._chunk_index.clear()
        self._vocabulary.clear()
        self._total_chunks = 0
        for entry_data in data.get("entries", []):
            entry_data["source"] = KnowledgeSource(entry_data.get("source", "custom"))
            entry = KnowledgeEntry(**entry_data)
            self._entries[entry.id] = entry
            self._index_entry(entry)
        logger.info("Loaded %d entries from %s", len(self._entries), self.storage_path)

    def stats(self) -> dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "total_chunks": self._total_chunks,
            "vocabulary_size": len(self._vocabulary),
            "by_source": {
                s.value: sum(1 for e in self._entries.values() if e.source == s)
                for s in KnowledgeSource
            },
            "categories": list({e.category for e in self._entries.values()}),
        }
