"""Workflow graph topology: DAG traversal for upstream/downstream queries.

Integrates Dify WorkflowGraphTopology pattern:
- WorkflowGraphTopology: graph structure with node/edge tracking
  - from_graph: build from nodes + edges mapping
  - has_node: check node existence
  - is_upstream: BFS check if source is upstream of target
  - upstream_node_ids: all nodes reachable upstream (transitive closure)
  - downstream_node_ids: all nodes reachable downstream
  - node_ids: set of all node IDs
  - incoming/outgoing: adjacency maps
  - roots: nodes with no incoming edges
  - leaves: nodes with no outgoing edges
  - topological_order: linear ordering respecting dependencies
  - has_cycle: cycle detection via Kahn's algorithm

Useful for workflow engines that need to determine execution order,
find dependencies, or validate graph structure.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


class WorkflowGraphTopology:
    """Graph topology with upstream/downstream traversal.

    Built from a graph mapping with 'nodes' (list of {id: ...}) and
    'edges' (list of {source: ..., target: ...}).
    """

    def __init__(
        self,
        *,
        node_ids: set[str],
        incoming: Mapping[str, Sequence[str]],
        outgoing: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._node_ids = set(node_ids)
        self._incoming: dict[str, list[str]] = defaultdict(list)
        self._outgoing: dict[str, list[str]] = defaultdict(list)
        for k, v in incoming.items():
            self._incoming[k] = list(v)
        if outgoing:
            for k, v in outgoing.items():
                self._outgoing[k] = list(v)
        else:
            # Build outgoing from incoming
            for target, sources in self._incoming.items():
                for source in sources:
                    self._outgoing[source].append(target)

    @classmethod
    def from_graph(cls, graph: Mapping[str, Any]) -> WorkflowGraphTopology:
        """Build topology from a graph mapping with 'nodes' and 'edges'."""
        node_ids = cls._node_ids_from_graph(graph)
        incoming: dict[str, list[str]] = defaultdict(list)
        outgoing: dict[str, list[str]] = defaultdict(list)
        edges = graph.get("edges")
        if isinstance(edges, list):
            for edge in edges:
                if not isinstance(edge, Mapping):
                    continue
                source = edge.get("source")
                target = edge.get("target")
                if isinstance(source, str) and isinstance(target, str):
                    incoming[target].append(source)
                    outgoing[source].append(target)
        return cls(node_ids=node_ids, incoming=incoming, outgoing=outgoing)

    @staticmethod
    def _node_ids_from_graph(graph: Mapping[str, Any]) -> set[str]:
        node_ids: set[str] = set()
        nodes = graph.get("nodes")
        if not isinstance(nodes, list):
            return node_ids
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_id = node.get("id")
            if isinstance(node_id, str):
                node_ids.add(node_id)
        return node_ids

    @property
    def node_ids(self) -> set[str]:
        """All node IDs in the graph."""
        return set(self._node_ids)

    @property
    def edge_count(self) -> int:
        """Total number of edges."""
        return sum(len(v) for v in self._outgoing.values())

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._node_ids

    def incoming_of(self, node_id: str) -> list[str]:
        """Direct predecessors of a node."""
        return list(self._incoming.get(node_id, []))

    def outgoing_of(self, node_id: str) -> list[str]:
        """Direct successors of a node."""
        return list(self._outgoing.get(node_id, []))

    def is_upstream(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
    ) -> bool:
        """Check if source_node_id is upstream of target_node_id (BFS)."""
        if source_node_id == target_node_id:
            return False
        visited: set[str] = set()
        queue: deque[str] = deque(self._incoming.get(target_node_id, ()))
        while queue:
            candidate = queue.popleft()
            if candidate == source_node_id:
                return True
            if candidate in visited:
                continue
            visited.add(candidate)
            queue.extend(self._incoming.get(candidate, ()))
        return False

    def is_downstream(
        self,
        *,
        source_node_id: str,
        target_node_id: str,
    ) -> bool:
        """Check if target_node_id is downstream of source_node_id.

        target is downstream of source iff there is a path source → target,
        which is the same as source being upstream of target.
        """
        return self.is_upstream(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
        )

    def upstream_node_ids(self, target_node_id: str) -> set[str]:
        """All nodes reachable upstream of target_node_id (excluding it).

        Edges may reference ids missing from nodes; only real nodes returned.
        """
        visited: set[str] = set()
        queue: deque[str] = deque(self._incoming.get(target_node_id, ()))
        while queue:
            candidate = queue.popleft()
            if candidate in visited:
                continue
            visited.add(candidate)
            queue.extend(self._incoming.get(candidate, ()))
        visited.discard(target_node_id)
        return visited & self._node_ids

    def downstream_node_ids(self, source_node_id: str) -> set[str]:
        """All nodes reachable downstream of source_node_id (excluding it)."""
        visited: set[str] = set()
        queue: deque[str] = deque(self._outgoing.get(source_node_id, ()))
        while queue:
            candidate = queue.popleft()
            if candidate in visited:
                continue
            visited.add(candidate)
            queue.extend(self._outgoing.get(candidate, ()))
        visited.discard(source_node_id)
        return visited & self._node_ids

    def roots(self) -> set[str]:
        """Nodes with no incoming edges (entry points)."""
        return {
            nid for nid in self._node_ids
            if not self._incoming.get(nid)
        }

    def leaves(self) -> set[str]:
        """Nodes with no outgoing edges (terminal nodes)."""
        return {
            nid for nid in self._node_ids
            if not self._outgoing.get(nid)
        }

    def topological_order(self) -> list[str] | None:
        """Return nodes in topological order, or None if cycle exists.

        Uses Kahn's algorithm: repeatedly remove nodes with no incoming
        edges. If not all nodes are removed, a cycle exists.
        Edges to nonexistent nodes are ignored.
        """
        in_degree: dict[str, int] = {}
        for nid in self._node_ids:
            in_degree[nid] = 0

        for nid in self._node_ids:
            for successor in self._outgoing.get(nid, []):
                if successor in in_degree:
                    in_degree[successor] += 1

        queue: deque[str] = deque(
            nid for nid in self._node_ids if in_degree[nid] == 0
        )
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for successor in self._outgoing.get(node, []):
                if successor in in_degree:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)

        if len(order) != len(self._node_ids):
            return None  # Cycle exists
        return order

    def has_cycle(self) -> bool:
        """Check if the graph contains a cycle."""
        return self.topological_order() is None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "node_ids": list(self._node_ids),
            "incoming": dict(self._incoming),
            "outgoing": dict(self._outgoing),
        }
