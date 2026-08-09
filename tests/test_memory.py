"""Tests for the memory system."""

import tempfile

import pytest

from daemon_engine.memory.code_memory import CodeMemory
from daemon_engine.memory.knowledge_memory import KnowledgeMemory
from daemon_engine.memory.long_term_memory import LongTermMemory, MemoryType
from daemon_engine.memory.unified import UnifiedMemory


class TestCodeMemory:
    def test_store_and_recall(self):
        mem = CodeMemory()
        mem.store(content="def hello(): print('hi')", language="python", file_path="test.py", artifact_type="function")
        result = mem.recall("hello function")
        assert "hello" in result.lower() or "hi" in result.lower()

    def test_search(self):
        mem = CodeMemory()
        mem.store(content="def add(a, b): return a + b", language="python")
        mem.store(content="class User: pass", language="python")
        results = mem.search("add function")
        assert len(results) > 0

    def test_get_by_file(self):
        mem = CodeMemory()
        mem.store(content="code1", file_path="file1.py")
        mem.store(content="code2", file_path="file1.py")
        results = mem.get_by_file("file1.py")
        assert len(results) == 2

    def test_get_by_tag(self):
        mem = CodeMemory()
        mem.store(content="code", tags=["utility", "helper"])
        results = mem.get_by_tag("utility")
        assert len(results) == 1

    def test_delete(self):
        mem = CodeMemory()
        artifact = mem.store(content="temp code")
        assert mem.delete(artifact.id) is True
        assert mem.get(artifact.id) is None

    def test_stats(self):
        mem = CodeMemory()
        mem.store(content="code1", language="python")
        mem.store(content="code2", language="javascript")
        stats = mem.stats()
        assert stats["total_artifacts"] == 2
        assert "python" in stats["languages"]

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem1 = CodeMemory(storage_path=tmpdir)
            mem1.store(content="persisted code", language="python")
            mem1.save()
            mem2 = CodeMemory(storage_path=tmpdir)
            assert len(mem2.search("persisted")) > 0


class TestKnowledgeMemory:
    def test_add_and_search(self):
        mem = KnowledgeMemory()
        mem.add_knowledge("Python", "A programming language", category="technology")
        results = mem.search("programming language")
        assert len(results) > 0

    def test_relate(self):
        mem = KnowledgeMemory()
        mem.add_knowledge("Python", "Programming language")
        mem.add_knowledge("Django", "Web framework")
        edge = mem.relate("Python", "Django", relation="has_framework")
        assert edge is not None

    def test_get_neighbors(self):
        mem = KnowledgeMemory()
        mem.add_knowledge("A", "Concept A")
        mem.add_knowledge("B", "Concept B")
        mem.relate("A", "B")
        neighbors = mem.get_neighbors("A")
        assert len(neighbors) == 1
        assert neighbors[0].concept == "B"

    def test_get_by_category(self):
        mem = KnowledgeMemory()
        mem.add_knowledge("X", "Content X", category="science")
        mem.add_knowledge("Y", "Content Y", category="science")
        results = mem.get_by_category("science")
        assert len(results) == 2

    def test_stats(self):
        mem = KnowledgeMemory()
        mem.add_knowledge("A", "Content A")
        stats = mem.stats()
        assert stats["total_nodes"] == 1


class TestLongTermMemory:
    def test_store_and_recall(self):
        mem = LongTermMemory()
        mem.store("I learned about Python today", memory_type=MemoryType.EPISODIC)
        result = mem.recall("Python")
        assert "Python" in result

    def test_search_by_type(self):
        mem = LongTermMemory()
        mem.store("Fact 1", memory_type=MemoryType.SEMANTIC)
        mem.store("Event 1", memory_type=MemoryType.EPISODIC)
        semantic = mem.get_by_type(MemoryType.SEMANTIC)
        assert len(semantic) == 1

    def test_reinforce(self):
        mem = LongTermMemory()
        record = mem.store("Important fact", importance=0.5)
        original = record.importance
        mem.reinforce(record.id)
        assert mem._records[record.id].importance > original

    def test_consolidate(self):
        mem = LongTermMemory()
        mem.store("Low importance", importance=0.02)
        pruned = mem.consolidate()
        assert isinstance(pruned, int)

    def test_stats(self):
        mem = LongTermMemory()
        mem.store("Memory 1", memory_type=MemoryType.EPISODIC)
        stats = mem.stats()
        assert stats["total_records"] == 1


class TestUnifiedMemory:
    def test_store_code(self):
        mem = UnifiedMemory()
        mem.store("def foo(): pass", memory_type="code", language="python")
        result = mem.recall("foo")
        assert result

    def test_store_knowledge(self):
        mem = UnifiedMemory()
        mem.store("Python is great", memory_type="knowledge", concept="Python")
        result = mem.recall("Python")
        assert result

    def test_store_episodic(self):
        mem = UnifiedMemory()
        mem.store("I solved a bug today", memory_type="episodic")
        result = mem.recall("bug")
        assert result

    def test_search_all(self):
        mem = UnifiedMemory()
        mem.store("code snippet", memory_type="code", language="python")
        mem.store("knowledge fact", memory_type="knowledge", concept="fact")
        results = mem.search_all("snippet")
        assert "code" in results
        assert "knowledge" in results

    def test_stats(self):
        mem = UnifiedMemory()
        mem.store("something", memory_type="episodic")
        stats = mem.stats()
        assert "code_memory" in stats
        assert "knowledge_memory" in stats
        assert "long_term_memory" in stats
