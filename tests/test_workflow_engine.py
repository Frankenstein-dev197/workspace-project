"""Tests for workflow execution engine."""

import threading
import time

import pytest

from daemon_engine.core.workflow_engine import (
    WorkflowEngine,
    WorkflowNode,
    NodeRunResult,
    NodeStatus,
)


class TestNodeStatus:
    def test_values(self):
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.RUNNING.value == "running"
        assert NodeStatus.SUCCEEDED.value == "succeeded"
        assert NodeStatus.FAILED.value == "failed"
        assert NodeStatus.SKIPPED.value == "skipped"


class TestNodeRunResult:
    def test_creation(self):
        result = NodeRunResult(node_id="a")
        assert result.node_id == "a"
        assert result.status == NodeStatus.SUCCEEDED

    def test_to_dict(self):
        result = NodeRunResult(
            node_id="a",
            status=NodeStatus.FAILED,
            error="something went wrong",
            execution_time=0.5,
        )
        d = result.to_dict()
        assert d["node_id"] == "a"
        assert d["status"] == "failed"
        assert d["error"] == "something went wrong"


class TestWorkflowNode:
    def test_creation(self):
        node = WorkflowNode(id="a", type="compute")
        assert node.id == "a"
        assert node.type == "compute"

    def test_run_no_runner(self):
        node = WorkflowNode(id="a")
        assert node.run({}, {}) == {}

    def test_run_with_runner(self):
        def runner(inputs, ctx):
            return {"result": inputs.get("x", 0) + 1}

        node = WorkflowNode(id="a", runner=runner)
        result = node.run({"x": 5}, {})
        assert result == {"result": 6}

    def test_to_dict(self):
        node = WorkflowNode(id="a", type="compute", config={"key": "value"})
        d = node.to_dict()
        assert d["id"] == "a"
        assert d["type"] == "compute"
        assert d["config"] == {"key": "value"}


class TestWorkflowEngine:
    def test_single_node(self):
        engine = WorkflowEngine()
        node = WorkflowNode(
            id="a",
            runner=lambda inputs, ctx: {"output": 42},
        )
        results = engine.execute([node], [])
        assert results["a"].status == NodeStatus.SUCCEEDED
        assert results["a"].outputs == {"output": 42}

    def test_linear_chain(self):
        engine = WorkflowEngine()
        nodes = [
            WorkflowNode(id="a", runner=lambda i, c: {"x": 1}),
            WorkflowNode(id="b", runner=lambda i, c: {"y": i.get("x", 0) + 1}),
            WorkflowNode(id="c", runner=lambda i, c: {"z": i.get("y", 0) + 1}),
        ]
        edges = [("a", "b"), ("b", "c")]
        results = engine.execute(nodes, edges)
        assert results["a"].outputs == {"x": 1}
        assert results["b"].outputs == {"y": 2}
        assert results["c"].outputs == {"z": 3}

    def test_parallel_independent_nodes(self):
        engine = WorkflowEngine(max_workers=4)
        nodes = [
            WorkflowNode(id="a", runner=lambda i, c: {"x": 1}),
            WorkflowNode(id="b", runner=lambda i, c: {"y": 2}),
            WorkflowNode(id="c", runner=lambda i, c: {"z": 3}),
        ]
        results = engine.execute(nodes, [])
        assert all(
            results[n.id].status == NodeStatus.SUCCEEDED for n in nodes
        )

    def test_diamond_pattern(self):
        # a -> b -> d
        # a -> c -> d
        engine = WorkflowEngine()
        nodes = [
            WorkflowNode(id="a", runner=lambda i, c: {"val": 10}),
            WorkflowNode(id="b", runner=lambda i, c: {"b": i.get("val", 0) * 2}),
            WorkflowNode(id="c", runner=lambda i, c: {"c": i.get("val", 0) * 3}),
            WorkflowNode(
                id="d",
                runner=lambda i, c: {"total": i.get("b", 0) + i.get("c", 0)},
            ),
        ]
        edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
        results = engine.execute(nodes, edges)
        assert results["d"].outputs == {"total": 20 + 30}

    def test_node_failure(self):
        engine = WorkflowEngine(stop_on_error=True)

        def failing_runner(inputs, ctx):
            raise ValueError("boom")

        nodes = [
            WorkflowNode(id="a", runner=failing_runner),
            WorkflowNode(id="b", runner=lambda i, c: {"y": 1}),
        ]
        edges = [("a", "b")]
        results = engine.execute(nodes, edges)
        assert results["a"].status == NodeStatus.FAILED
        assert "boom" in results["a"].error
        assert results["b"].status == NodeStatus.SKIPPED

    def test_stop_on_error_false(self):
        engine = WorkflowEngine(stop_on_error=False)

        def failing_runner(inputs, ctx):
            raise ValueError("boom")

        nodes = [
            WorkflowNode(id="a", runner=failing_runner),
            WorkflowNode(id="b", runner=lambda i, c: {"y": 1}),
        ]
        # b has no dependency on a
        results = engine.execute(nodes, [])
        assert results["a"].status == NodeStatus.FAILED
        assert results["b"].status == NodeStatus.SUCCEEDED

    def test_cycle_detected(self):
        engine = WorkflowEngine()
        nodes = [
            WorkflowNode(id="a", runner=lambda i, c: {}),
            WorkflowNode(id="b", runner=lambda i, c: {}),
            WorkflowNode(id="c", runner=lambda i, c: {}),
        ]
        edges = [("a", "b"), ("b", "c"), ("c", "a")]
        results = engine.execute(nodes, edges)
        assert all(
            results[n.id].status == NodeStatus.FAILED for n in nodes
        )
        assert all(
            "cycle" in results[n.id].error.lower() for n in nodes
        )

    def test_initial_context_passed(self):
        engine = WorkflowEngine()
        node = WorkflowNode(
            id="a",
            runner=lambda i, c: {"result": c.get("key", "default")},
        )
        results = engine.execute([node], [], initial_context={"key": "value"})
        assert results["a"].outputs == {"result": "value"}

    def test_upstream_outputs_merged(self):
        engine = WorkflowEngine()
        nodes = [
            WorkflowNode(id="a", runner=lambda i, c: {"a_val": 1}),
            WorkflowNode(id="b", runner=lambda i, c: {"b_val": 2}),
            WorkflowNode(
                id="c",
                runner=lambda i, c: {
                    "sum": i.get("a_val", 0) + i.get("b_val", 0)
                },
            ),
        ]
        edges = [("a", "c"), ("b", "c")]
        results = engine.execute(nodes, edges)
        assert results["c"].outputs == {"sum": 3}

    def test_execution_time_recorded(self):
        engine = WorkflowEngine()

        def slow_runner(inputs, ctx):
            time.sleep(0.05)
            return {"done": True}

        node = WorkflowNode(id="a", runner=slow_runner)
        results = engine.execute([node], [])
        assert results["a"].execution_time >= 0.04

    def test_empty_workflow(self):
        engine = WorkflowEngine()
        results = engine.execute([], [])
        assert results == {}

    def test_node_not_found(self):
        engine = WorkflowEngine()
        # Edge references nonexistent node, but that node isn't in nodes list
        nodes = [WorkflowNode(id="a", runner=lambda i, c: {"x": 1})]
        edges = [("a", "nonexistent")]
        results = engine.execute(nodes, edges)
        # 'a' should still execute successfully
        assert "a" in results
        assert results["a"].status == NodeStatus.SUCCEEDED

    def test_runner_returns_non_dict(self):
        engine = WorkflowEngine()
        node = WorkflowNode(
            id="a",
            runner=lambda i, c: "not a dict",
        )
        results = engine.execute([node], [])
        assert results["a"].status == NodeStatus.SUCCEEDED
        assert results["a"].outputs == {"result": "not a dict"}


class TestExecuteSummary:
    def test_summary_all_success(self):
        engine = WorkflowEngine()
        nodes = [
            WorkflowNode(id="a", runner=lambda i, c: {"x": 1}),
            WorkflowNode(id="b", runner=lambda i, c: {"y": 2}),
        ]
        results = engine.execute(nodes, [])
        summary = engine.execute_summary(results)
        assert summary["total_nodes"] == 2
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0
        assert summary["skipped"] == 0

    def test_summary_with_failure(self):
        engine = WorkflowEngine()
        nodes = [
            WorkflowNode(
                id="a",
                runner=lambda i, c: (_ for _ in ()).throw(ValueError("err")),
            ),
            WorkflowNode(id="b", runner=lambda i, c: {"y": 2}),
        ]
        results = engine.execute(nodes, [])
        summary = engine.execute_summary(results)
        assert summary["failed"] >= 1

    def test_summary_includes_results(self):
        engine = WorkflowEngine()
        node = WorkflowNode(id="a", runner=lambda i, c: {"x": 1})
        results = engine.execute([node], [])
        summary = engine.execute_summary(results)
        assert "results" in summary
        assert "a" in summary["results"]


class TestThreadSafety:
    def test_concurrent_node_execution(self):
        engine = WorkflowEngine(max_workers=4)
        counter = {"count": 0}
        lock = threading.Lock()

        def incrementing_runner(inputs, ctx):
            with lock:
                counter["count"] += 1
            return {"n": counter["count"]}

        nodes = [
            WorkflowNode(id=f"node-{i}", runner=incrementing_runner)
            for i in range(10)
        ]
        results = engine.execute(nodes, [])
        assert len(results) == 10
        assert all(
            r.status == NodeStatus.SUCCEEDED for r in results.values()
        )
        assert counter["count"] == 10
