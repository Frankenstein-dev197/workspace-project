"""Tests for the knowledge base system."""

import pytest

from daemon_engine.knowledge.knowledge_base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeSource,
)
from daemon_engine.knowledge.algorithm_patterns import AlgorithmPatternLibrary
from daemon_engine.knowledge.devops_knowledge import DevOpsKnowledgeBase


class TestKnowledgeBase:
    def test_add_entry(self):
        kb = KnowledgeBase()
        entry = kb.add_entry("Test", "Some content about testing", tags=["test"])
        assert entry.title == "Test"
        assert entry.id in kb.all_entries().__repr__() or kb.get_entry(entry.id) is not None

    def test_search(self):
        kb = KnowledgeBase()
        kb.add_entry("Python Guide", "Python is a programming language", tags=["python"])
        kb.add_entry("Docker Guide", "Docker is a containerization tool", tags=["docker"])
        results = kb.search("python")
        assert len(results) > 0
        assert "Python" in results[0][1].title

    def test_chunking(self):
        kb = KnowledgeBase(chunk_size=100, chunk_overlap=20)
        long_content = "A" * 250 + "\n\n" + "B" * 100
        entry = kb.add_entry("Long", long_content)
        assert len(entry.chunks) > 1

    def test_get_by_tag(self):
        kb = KnowledgeBase()
        kb.add_entry("Test1", "Content1", tags=["python", "testing"])
        kb.add_entry("Test2", "Content2", tags=["docker"])
        results = kb.get_by_tag("python")
        assert len(results) == 1

    def test_get_by_source(self):
        kb = KnowledgeBase()
        kb.add_entry("Test1", "Content1", source=KnowledgeSource.LEETCODE)
        kb.add_entry("Test2", "Content2", source=KnowledgeSource.DEVOPS)
        leetcode = kb.get_by_source(KnowledgeSource.LEETCODE)
        assert len(leetcode) == 1

    def test_remove_entry(self):
        kb = KnowledgeBase()
        entry = kb.add_entry("Test", "Content")
        assert kb.remove_entry(entry.id) is True
        assert kb.get_entry(entry.id) is None

    def test_stats(self):
        kb = KnowledgeBase()
        kb.add_entry("Test1", "Content1", source=KnowledgeSource.CUSTOM)
        kb.add_entry("Test2", "Content2", source=KnowledgeSource.LEETCODE)
        stats = kb.stats()
        assert stats["total_entries"] == 2
        assert stats["by_source"]["custom"] == 1

    def test_save_and_load(self, tmp_path):
        kb = KnowledgeBase(storage_path=tmp_path)
        kb.add_entry("Test", "Some content for persistence")
        kb.save()
        kb2 = KnowledgeBase(storage_path=tmp_path)
        kb2.load()
        assert kb2.stats()["total_entries"] == 1


class TestAlgorithmPatternLibrary:
    def test_load_patterns(self):
        lib = AlgorithmPatternLibrary()
        count = lib.load_patterns()
        assert count == 10

    def test_search_patterns(self):
        lib = AlgorithmPatternLibrary()
        lib.load_patterns()
        results = lib.search_patterns("binary search")
        assert len(results) > 0

    def test_get_pattern(self):
        lib = AlgorithmPatternLibrary()
        lib.load_patterns()
        pattern = lib.get_pattern("Dynamic")
        assert pattern is not None
        assert "Dynamic" in pattern.title

    def test_list_patterns(self):
        lib = AlgorithmPatternLibrary()
        lib.load_patterns()
        patterns = lib.list_patterns()
        assert len(patterns) == 10
        assert "Binary Search" in patterns


class TestDevOpsKnowledgeBase:
    def test_load_knowledge(self):
        kb = DevOpsKnowledgeBase()
        count = kb.load_knowledge()
        assert count == 8

    def test_search(self):
        kb = DevOpsKnowledgeBase()
        kb.load_knowledge()
        results = kb.search("docker")
        assert len(results) > 0

    def test_get_topic(self):
        kb = DevOpsKnowledgeBase()
        kb.load_knowledge()
        topic = kb.get_topic("Kubernetes")
        assert topic is not None

    def test_list_topics(self):
        kb = DevOpsKnowledgeBase()
        kb.load_knowledge()
        topics = kb.list_topics()
        assert len(topics) == 8
