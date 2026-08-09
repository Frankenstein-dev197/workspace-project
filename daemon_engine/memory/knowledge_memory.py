"""Knowledge Memory: structured knowledge graph storage.

Inspired by Headroom's SQLite graph store and Reference's knowledge base.
Stores facts, concepts, and relationships as a lightweight graph with
semantic retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    id: str = ""
    concept: str = ""
    content: str = ""
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha256(f"{self.concept}:{self.content}".encode()).hexdigest()[:16]


@dataclass
class KnowledgeEdge:
    source_id: str = ""
    target_id: str = ""
    relation: str = "related_to"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeMemory:
    """Graph-based knowledge store with semantic search and relationship traversal."""

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: list[KnowledgeEdge] = []
        self._edge_index: dict[str, list[KnowledgeEdge]] = {}
        if self.storage_path:
            self._load()

    def add_knowledge(
        self,
        concept: str,
        content: str,
        category: str = "general",
        tags: list[str] | None = None,
        source: str = "",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeNode:
        node = KnowledgeNode(
            concept=concept,
            content=content,
            category=category,
            tags=tags or [],
            source=source,
            confidence=confidence,
            metadata=metadata or {},
            embedding=self._embed(f"{concept} {content}"),
        )
        self._nodes[node.id] = node
        logger.debug("Added knowledge node: %s (%s)", concept, node.id)
        return node

    def relate(
        self,
        source_concept: str,
        target_concept: str,
        relation: str = "related_to",
        weight: float = 1.0,
    ) -> KnowledgeEdge | None:
        source = self._find_by_concept(source_concept)
        target = self._find_by_concept(target_concept)
        if not source or not target:
            logger.warning("Cannot relate: missing node(s) %s -> %s", source_concept, target_concept)
            return None
        edge = KnowledgeEdge(
            source_id=source.id,
            target_id=target.id,
            relation=relation,
            weight=weight,
        )
        self._edges.append(edge)
        self._edge_index.setdefault(source.id, []).append(edge)
        return edge

    def recall(self, query: str, limit: int = 5) -> str:
        results = self.search(query, limit=limit)
        if not results:
            return ""
        parts: list[str] = []
        for node in results:
            parts.append(f"[{node.category}] {node.concept}: {node.content[:300]}")
        return "\n---\n".join(parts)

    def search(self, query: str, limit: int = 5) -> list[KnowledgeNode]:
        query_emb = self._embed(query)
        scored: list[tuple[float, KnowledgeNode]] = []
        for node in self._nodes.values():
            sim = self._cosine_sim(query_emb, node.embedding)
            scored.append((sim * node.confidence, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[KnowledgeNode] = []
        for _, node in scored[:limit]:
            node.accessed_at = time.time()
            results.append(node)
        return results

    def get_neighbors(self, concept: str, depth: int = 1) -> list[KnowledgeNode]:
        node = self._find_by_concept(concept)
        if not node:
            return []
        visited: set[str] = {node.id}
        result: list[KnowledgeNode] = []
        frontier = [node.id]
        for _ in range(depth):
            next_frontier: list[str] = []
            for nid in frontier:
                for edge in self._edge_index.get(nid, []):
                    if edge.target_id not in visited:
                        visited.add(edge.target_id)
                        neighbor = self._nodes.get(edge.target_id)
                        if neighbor:
                            result.append(neighbor)
                            next_frontier.append(edge.target_id)
            frontier = next_frontier
        return result

    def get_by_category(self, category: str) -> list[KnowledgeNode]:
        return [n for n in self._nodes.values() if n.category == category]

    def get_by_tag(self, tag: str) -> list[KnowledgeNode]:
        return [n for n in self._nodes.values() if tag in n.tags]

    def _find_by_concept(self, concept: str) -> KnowledgeNode | None:
        for node in self._nodes.values():
            if node.concept.lower() == concept.lower():
                return node
        return None

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
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "categories": list({n.category for n in self._nodes.values()}),
            "tags": list({t for n in self._nodes.values() for t in n.tags}),
        }

    def _load(self) -> None:
        if not self.storage_path:
            return
        path = self.storage_path / "knowledge_memory.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            for item in data.get("nodes", []):
                node = KnowledgeNode(**item)
                self._nodes[node.id] = node
            for item in data.get("edges", []):
                edge = KnowledgeEdge(**item)
                self._edges.append(edge)
                self._edge_index.setdefault(edge.source_id, []).append(edge)
            logger.info("Loaded %d nodes, %d edges", len(self._nodes), len(self._edges))
        except Exception as exc:
            logger.error("Failed to load knowledge memory: %s", exc)

    def save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        path = self.storage_path / "knowledge_memory.json"
        data = {
            "nodes": [asdict(n) for n in self._nodes.values()],
            "edges": [asdict(e) for e in self._edges],
        }
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved knowledge memory: %d nodes, %d edges", len(self._nodes), len(self._edges))

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._edge_index.clear()
