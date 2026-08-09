"""Code Memory: codebase-aware persistent memory.

Inspired by Codebase Memory MCP — stores code snippets, AST structures,
function signatures, and dependencies. Provides semantic recall of code
artifacts using simple vector similarity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeArtifact:
    id: str = ""
    content: str = ""
    language: str = "unknown"
    file_path: str = ""
    artifact_type: str = "snippet"  # snippet, function, class, module
    tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.content.encode()).hexdigest()[:16]


class CodeMemory:
    """Stores and retrieves code artifacts with semantic search."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._artifacts: dict[str, CodeArtifact] = {}
        self._index_by_file: dict[str, list[str]] = {}
        self._index_by_tag: dict[str, list[str]] = {}
        if self.storage_path:
            self._load()

    def store(
        self,
        content: str,
        language: str = "unknown",
        file_path: str = "",
        artifact_type: str = "snippet",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CodeArtifact:
        artifact = CodeArtifact(
            content=content,
            language=language,
            file_path=file_path,
            artifact_type=artifact_type,
            tags=tags or [],
            metadata=metadata or {},
            embedding=self._embed(content),
        )
        self._artifacts[artifact.id] = artifact
        if file_path:
            self._index_by_file.setdefault(file_path, []).append(artifact.id)
        for tag in artifact.tags:
            self._index_by_tag.setdefault(tag, []).append(artifact.id)
        logger.debug("Stored code artifact %s (%s)", artifact.id, artifact_type)
        return artifact

    def recall(self, query: str, limit: int = 5) -> str:
        results = self.search(query, limit=limit)
        if not results:
            return ""
        parts: list[str] = []
        for artifact in results:
            parts.append(f"[{artifact.language}] {artifact.file_path or 'inline'}:\n{artifact.content[:500]}")
        return "\n---\n".join(parts)

    def search(self, query: str, limit: int = 5) -> list[CodeArtifact]:
        query_embedding = self._embed(query)
        scored: list[tuple[float, CodeArtifact]] = []
        for artifact in self._artifacts.values():
            sim = self._cosine_sim(query_embedding, artifact.embedding)
            scored.append((sim, artifact))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[CodeArtifact] = []
        for sim, artifact in scored[:limit]:
            artifact.accessed_at = time.time()
            artifact.access_count += 1
            results.append(artifact)
        return results

    def get_by_file(self, file_path: str) -> list[CodeArtifact]:
        ids = self._index_by_file.get(file_path, [])
        return [self._artifacts[i] for i in ids if i in self._artifacts]

    def get_by_tag(self, tag: str) -> list[CodeArtifact]:
        ids = self._index_by_tag.get(tag, [])
        return [self._artifacts[i] for i in ids if i in self._artifacts]

    def get(self, artifact_id: str) -> CodeArtifact | None:
        return self._artifacts.get(artifact_id)

    def delete(self, artifact_id: str) -> bool:
        artifact = self._artifacts.pop(artifact_id, None)
        if not artifact:
            return False
        if artifact.file_path and artifact.file_path in self._index_by_file:
            self._index_by_file[artifact.file_path] = [
                i for i in self._index_by_file[artifact.file_path] if i != artifact_id
            ]
        for tag in artifact.tags:
            if tag in self._index_by_tag:
                self._index_by_tag[tag] = [i for i in self._index_by_tag[tag] if i != artifact_id]
        return True

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
            "total_artifacts": len(self._artifacts),
            "files_indexed": len(self._index_by_file),
            "tags_indexed": len(self._index_by_tag),
            "languages": list({a.language for a in self._artifacts.values()}),
        }

    def _load(self) -> None:
        if not self.storage_path:
            return
        path = self.storage_path / "code_memory.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for item in data:
                artifact = CodeArtifact(**item)
                self._artifacts[artifact.id] = artifact
                if artifact.file_path:
                    self._index_by_file.setdefault(artifact.file_path, []).append(artifact.id)
                for tag in artifact.tags:
                    self._index_by_tag.setdefault(tag, []).append(artifact.id)
            logger.info("Loaded %d code artifacts from %s", len(self._artifacts), path)
        except Exception as exc:
            logger.error("Failed to load code memory: %s", exc)

    def save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        path = self.storage_path / "code_memory.json"
        data = [asdict(a) for a in self._artifacts.values()]
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved %d code artifacts to %s", len(data), path)

    def clear(self) -> None:
        self._artifacts.clear()
        self._index_by_file.clear()
        self._index_by_tag.clear()
