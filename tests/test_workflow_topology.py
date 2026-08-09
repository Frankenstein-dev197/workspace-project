"""Tests for workflow graph topology."""

import pytest

from daemon_engine.core.workflow_topology import WorkflowGraphTopology


class TestFromGraph:
    def test_simple_graph(self):
        graph = {
            "nodes": [
                {"id": "a"},
                {"id": "b"},
                {"id": "c"},
            ],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.has_node("a")
        assert topo.has_node("b")
        assert topo.has_node("c")
        assert not topo.has_node("d")

    def test_empty_graph(self):
        topo = WorkflowGraphTopology.from_graph({})
        assert topo.node_ids == set()

    def test_nodes_only(self):
        graph = {"nodes": [{"id": "a"}, {"id": "b"}]}
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.node_ids == {"a", "b"}
        assert topo.edge_count == 0

    def test_edges_without_nodes_ignored_in_node_set(self):
        graph = {
            "nodes": [{"id": "a"}],
            "edges": [{"source": "a", "target": "b"}],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.has_node("a")
        assert not topo.has_node("b")  # b not in nodes

    def test_invalid_edge_entries_skipped(self):
        graph = {
            "nodes": [{"id": "a"}],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": 123, "target": "c"},  # invalid
                "not-a-dict",  # invalid
                {"source": "a"},  # missing target
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.has_node("a")


class TestHasNode:
    def test_existing(self):
        topo = WorkflowGraphTopology.from_graph({
            "nodes": [{"id": "x"}],
        })
        assert topo.has_node("x") is True

    def test_missing(self):
        topo = WorkflowGraphTopology.from_graph({
            "nodes": [{"id": "x"}],
        })
        assert topo.has_node("y") is False


class TestIsUpstream:
    def setup_method(self):
        # Graph: a -> b -> c -> d
        #              b -> e
        self.graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c", "d", "e"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "d"},
                {"source": "b", "target": "e"},
            ],
        }
        self.topo = WorkflowGraphTopology.from_graph(self.graph)

    def test_direct_upstream(self):
        assert self.topo.is_upstream(source_node_id="a", target_node_id="b") is True

    def test_transitive_upstream(self):
        assert self.topo.is_upstream(source_node_id="a", target_node_id="d") is True

    def test_not_upstream(self):
        assert self.topo.is_upstream(source_node_id="d", target_node_id="a") is False

    def test_same_node(self):
        assert self.topo.is_upstream(source_node_id="a", target_node_id="a") is False

    def test_unrelated_nodes(self):
        assert self.topo.is_upstream(source_node_id="d", target_node_id="e") is False

    def test_branch_upstream(self):
        # e is downstream of b, a is upstream of b
        assert self.topo.is_upstream(source_node_id="a", target_node_id="e") is True


class TestIsDownstream:
    def setup_method(self):
        self.graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ],
        }
        self.topo = WorkflowGraphTopology.from_graph(self.graph)

    def test_downstream(self):
        assert self.topo.is_downstream(source_node_id="a", target_node_id="c") is True

    def test_not_downstream(self):
        assert self.topo.is_downstream(source_node_id="c", target_node_id="a") is False


class TestUpstreamNodeIds:
    def setup_method(self):
        self.graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c", "d", "e"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "d"},
                {"source": "a", "target": "e"},
                {"source": "e", "target": "d"},
            ],
        }
        self.topo = WorkflowGraphTopology.from_graph(self.graph)

    def test_all_upstream(self):
        upstream = self.topo.upstream_node_ids("d")
        assert upstream == {"a", "b", "c", "e"}

    def test_excludes_self(self):
        upstream = self.topo.upstream_node_ids("a")
        assert "a" not in upstream

    def test_no_upstream(self):
        upstream = self.topo.upstream_node_ids("a")
        assert upstream == set()

    def test_filters_nonexistent_nodes(self):
        # Edge to nonexistent node should not appear
        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.upstream_node_ids("b") == {"a"}


class TestDownstreamNodeIds:
    def setup_method(self):
        self.graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c", "d"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "a", "target": "d"},
            ],
        }
        self.topo = WorkflowGraphTopology.from_graph(self.graph)

    def test_all_downstream(self):
        downstream = self.topo.downstream_node_ids("a")
        assert downstream == {"b", "c", "d"}

    def test_excludes_self(self):
        downstream = self.topo.downstream_node_ids("a")
        assert "a" not in downstream

    def test_no_downstream(self):
        downstream = self.topo.downstream_node_ids("c")
        assert downstream == set()


class TestRootsAndLeaves:
    def test_roots(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c", "d"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "d"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.roots() == {"a"}

    def test_multiple_roots(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c"]],
            "edges": [
                {"source": "a", "target": "c"},
                {"source": "b", "target": "c"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.roots() == {"a", "b"}

    def test_leaves(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c", "d"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "d"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.leaves() == {"d"}

    def test_multiple_leaves(self):
        nodes = [{"id": n} for n in ["a", "b", "c"]]
        graph = {
            "nodes": nodes,
            "edges": [{"source": "a", "target": "b"}, {"source": "a", "target": "c"}],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.leaves() == {"b", "c"}

    def test_isolated_nodes_are_both_roots_and_leaves(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b"]],
            "edges": [],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.roots() == {"a", "b"}
        assert topo.leaves() == {"a", "b"}


class TestTopologicalOrder:
    def test_linear_chain(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c", "d"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "d"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        order = topo.topological_order()
        assert order == ["a", "b", "c", "d"]

    def test_diamond(self):
        # a -> b -> d
        # a -> c -> d
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c", "d"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "a", "target": "c"},
                {"source": "b", "target": "d"},
                {"source": "c", "target": "d"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        order = topo.topological_order()
        assert order is not None
        assert order[0] == "a"
        assert order[-1] == "d"
        # b and c can be in either order
        assert set(order[1:3]) == {"b", "c"}

    def test_cycle_returns_none(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "a"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.topological_order() is None

    def test_empty_graph(self):
        topo = WorkflowGraphTopology.from_graph({})
        assert topo.topological_order() == []

    def test_single_node(self):
        graph = {"nodes": [{"id": "a"}], "edges": []}
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.topological_order() == ["a"]


class TestHasCycle:
    def test_no_cycle(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b"]],
            "edges": [{"source": "a", "target": "b"}],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.has_cycle() is False

    def test_has_cycle(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "c", "target": "a"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.has_cycle() is True

    def test_self_loop_is_cycle(self):
        graph = {
            "nodes": [{"id": "a"}],
            "edges": [{"source": "a", "target": "a"}],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.has_cycle() is True


class TestIncomingOutgoing:
    def test_incoming(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c"]],
            "edges": [
                {"source": "a", "target": "c"},
                {"source": "b", "target": "c"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert set(topo.incoming_of("c")) == {"a", "b"}
        assert topo.incoming_of("a") == []

    def test_outgoing(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "a", "target": "c"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert set(topo.outgoing_of("a")) == {"b", "c"}
        assert topo.outgoing_of("b") == []


class TestEdgeCount:
    def test_count(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b", "c"]],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.edge_count == 2

    def test_no_edges(self):
        graph = {"nodes": [{"id": "a"}]}
        topo = WorkflowGraphTopology.from_graph(graph)
        assert topo.edge_count == 0


class TestToDict:
    def test_serialization(self):
        graph = {
            "nodes": [{"id": n} for n in ["a", "b"]],
            "edges": [{"source": "a", "target": "b"}],
        }
        topo = WorkflowGraphTopology.from_graph(graph)
        d = topo.to_dict()
        assert "node_ids" in d
        assert "incoming" in d
        assert "outgoing" in d
        assert set(d["node_ids"]) == {"a", "b"}
