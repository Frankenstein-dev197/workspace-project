"""Workflow execution engine: DAG-based node execution with topological order.

Combines Dify workflow topology + node execution patterns:
- WorkflowNode: abstract node with id, type, config, and run() method
- NodeRunResult: result of node execution (status, outputs, error)
- NodeStatus: PENDING, RUNNING, SUCCEEDED, FAILED, SKIPPED
- WorkflowEngine: executes nodes in topological order
  - Built on WorkflowGraphTopology for dependency resolution
  - Executes nodes respecting dependencies (roots first)
  - Passes outputs from upstream nodes to downstream nodes
  - Supports parallel execution of independent nodes (same level)
  - Handles failures with configurable stop-on-error
  - Tracks execution state (running, completed, failed nodes)
  - Produces execution summary with per-node results

Useful for orchestrating multi-step agent workflows where each step
depends on outputs from previous steps.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from daemon_engine.core.workflow_topology import WorkflowGraphTopology

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a workflow node."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeRunResult:
    """Result of executing a workflow node."""
    node_id: str
    status: NodeStatus = NodeStatus.SUCCEEDED
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "outputs": self.outputs,
            "error": self.error,
            "execution_time": round(self.execution_time, 4),
        }


@dataclass
class WorkflowNode:
    """A workflow node definition.

    The `runner` callable receives (inputs, context) and returns a dict
    of outputs. `inputs` is the merged outputs of all upstream nodes.
    """
    id: str
    type: str = "default"
    config: dict[str, Any] = field(default_factory=dict)
    runner: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None

    def run(
        self,
        inputs: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute this node's runner."""
        if self.runner is None:
            return {}
        return self.runner(inputs, context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "config": self.config,
        }


class WorkflowEngine:
    """Executes workflow nodes in topological order.

    Supports parallel execution of independent nodes at the same
    dependency level, with configurable thread pool and error handling.
    """

    def __init__(
        self,
        *,
        max_workers: int = 4,
        stop_on_error: bool = True,
    ) -> None:
        self.max_workers = max_workers
        self.stop_on_error = stop_on_error
        self._lock = threading.Lock()

    def execute(
        self,
        nodes: list[WorkflowNode],
        edges: list[tuple[str, str]],
        *,
        initial_context: dict[str, Any] | None = None,
    ) -> dict[str, NodeRunResult]:
        """Execute a workflow graph.

        Args:
            nodes: list of workflow nodes
            edges: list of (source_id, target_id) edges
            initial_context: shared context dict for all nodes

        Returns:
            dict mapping node_id → NodeRunResult
        """
        node_map = {n.id: n for n in nodes}
        graph = self._build_graph(nodes, edges)
        topo = WorkflowGraphTopology.from_graph(graph)

        # Check for cycles
        if topo.has_cycle():
            failed = {}
            for node in nodes:
                failed[node.id] = NodeRunResult(
                    node_id=node.id,
                    status=NodeStatus.FAILED,
                    error="Graph contains a cycle",
                )
            return failed

        order = topo.topological_order() or []
        context = dict(initial_context or {})
        results: dict[str, NodeRunResult] = {}

        # Group nodes by dependency level for parallel execution
        levels = self._compute_levels(topo, order)

        for level_nodes in levels:
            level_results = self._execute_level(
                level_nodes, node_map, topo, context, results
            )
            results.update(level_results)

            if self.stop_on_error:
                has_error = any(
                    r.status == NodeStatus.FAILED for r in level_results.values()
                )
                if has_error:
                    # Mark remaining nodes as skipped
                    for node_id in order:
                        if node_id not in results:
                            results[node_id] = NodeRunResult(
                                node_id=node_id,
                                status=NodeStatus.SKIPPED,
                                error="Skipped due to upstream failure",
                            )
                    break

        return results

    def _build_graph(
        self,
        nodes: list[WorkflowNode],
        edges: list[tuple[str, str]],
    ) -> dict[str, Any]:
        """Build graph dict for WorkflowGraphTopology."""
        return {
            "nodes": [dict(id=n.id) for n in nodes],
            "edges": [
                {"source": s, "target": t} for s, t in edges
            ],
        }

    def _compute_levels(
        self,
        topo: WorkflowGraphTopology,
        order: list[str],
    ) -> list[list[str]]:
        """Group nodes into execution levels (same level = independent)."""
        if not order:
            return []

        levels: list[list[str]] = []
        completed: set[str] = set()

        remaining = list(order)
        while remaining:
            current_level = []
            for node_id in remaining:
                deps = topo.incoming_of(node_id)
                if all(d in completed for d in deps):
                    current_level.append(node_id)
            if not current_level:
                # Shouldn't happen if no cycle, but safety
                current_level = [remaining[0]]
            levels.append(current_level)
            completed.update(current_level)
            remaining = [n for n in remaining if n not in completed]

        return levels

    def _execute_level(
        self,
        level_nodes: list[str],
        node_map: dict[str, WorkflowNode],
        topo: WorkflowGraphTopology,
        context: dict[str, Any],
        prior_results: dict[str, NodeRunResult],
    ) -> dict[str, NodeRunResult]:
        """Execute all nodes at one level (potentially in parallel)."""
        if len(level_nodes) == 1:
            return {
                level_nodes[0]: self._execute_node(
                    level_nodes[0], node_map, topo, context, prior_results
                )
            }

        results: dict[str, NodeRunResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._execute_node,
                    node_id, node_map, topo, context, prior_results,
                ): node_id
                for node_id in level_nodes
            }
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    results[node_id] = future.result()
                except Exception as e:
                    results[node_id] = NodeRunResult(
                        node_id=node_id,
                        status=NodeStatus.FAILED,
                        error=str(e),
                    )
        return results

    def _execute_node(
        self,
        node_id: str,
        node_map: dict[str, WorkflowNode],
        topo: WorkflowGraphTopology,
        context: dict[str, Any],
        prior_results: dict[str, NodeRunResult],
    ) -> NodeRunResult:
        """Execute a single node, merging upstream outputs."""
        node = node_map.get(node_id)
        if node is None:
            return NodeRunResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                error=f"Node {node_id} not found",
            )

        # Check if any upstream failed
        upstream_ids = topo.incoming_of(node_id)
        upstream_failed = [
            uid for uid in upstream_ids
            if prior_results.get(uid, NodeRunResult(uid, NodeStatus.FAILED)).status
            == NodeStatus.FAILED
        ]
        if upstream_failed:
            return NodeRunResult(
                node_id=node_id,
                status=NodeStatus.SKIPPED,
                error=f"Upstream nodes failed: {upstream_failed}",
            )

        # Merge upstream outputs
        inputs: dict[str, Any] = {}
        for uid in upstream_ids:
            upstream_result = prior_results.get(uid)
            if upstream_result and upstream_result.outputs:
                inputs.update(upstream_result.outputs)

        # Execute
        start = time.time()
        try:
            outputs = node.run(inputs, context)
            elapsed = time.time() - start
            return NodeRunResult(
                node_id=node_id,
                status=NodeStatus.SUCCEEDED,
                outputs=outputs if isinstance(outputs, dict) else {"result": outputs},
                execution_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.exception(f"Node {node_id} failed: {e}")
            return NodeRunResult(
                node_id=node_id,
                status=NodeStatus.FAILED,
                error=str(e),
                execution_time=elapsed,
            )

    def execute_summary(
        self,
        results: dict[str, NodeRunResult],
    ) -> dict[str, Any]:
        """Produce a summary of execution results."""
        total = len(results)
        succeeded = sum(
            1 for r in results.values() if r.status == NodeStatus.SUCCEEDED
        )
        failed = sum(
            1 for r in results.values() if r.status == NodeStatus.FAILED
        )
        skipped = sum(
            1 for r in results.values() if r.status == NodeStatus.SKIPPED
        )
        total_time = sum(r.execution_time for r in results.values())
        return {
            "total_nodes": total,
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total_execution_time": round(total_time, 4),
            "results": {k: v.to_dict() for k, v in results.items()},
        }
