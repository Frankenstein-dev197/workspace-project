"""Tests for enhanced memory recall system."""

import pytest

from daemon_engine.memory.recall import (
    MemoryStore,
    MemoryEntry,
    MemoryType,
    FactScope,
    Freshness,
    RecallHit,
    RecallResult,
    tokenize,
    extract_keywords,
    compute_tfidf_score,
    assess_remember_write,
)


class TestTokenization:
    def test_tokenize(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_tokenize_empty(self):
        assert tokenize("") == []

    def test_tokenize_numbers(self):
        assert tokenize("Python 3.12") == ["python", "3", "12"]

    def test_extract_keywords(self):
        keywords = extract_keywords("The Python programming language is great for data science")
        assert "python" in keywords
        assert "the" not in keywords

    def test_extract_keywords_empty(self):
        assert extract_keywords("") == []

    def test_extract_keywords_max(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        keywords = extract_keywords(text, max_keywords=3)
        assert len(keywords) <= 3


class TestTFIDF:
    def test_score_with_match(self):
        docs = ["python programming", "java programming", "rust development"]
        score = compute_tfidf_score(["python"], "python programming", docs)
        assert score > 0

    def test_score_no_match(self):
        docs = ["python programming"]
        score = compute_tfidf_score(["java"], "python programming", docs)
        assert score == 0.0

    def test_score_empty_query(self):
        score = compute_tfidf_score([], "some text", ["some text"])
        assert score == 0.0


class TestMemoryEntry:
    def test_freshness_fresh(self):
        entry = MemoryEntry(id="1", name="test", description="desc", body="body")
        assert entry.freshness == Freshness.FRESH

    def test_freshness_stale(self):
        import time
        entry = MemoryEntry(
            id="1", name="test", description="desc", body="body",
            created_at=time.time() - 200 * 3600,
            updated_at=time.time() - 200 * 3600,
        )
        assert entry.freshness == Freshness.STALE

    def test_to_dict(self):
        entry = MemoryEntry(id="1", name="test", description="desc", body="body")
        d = entry.to_dict()
        assert d["id"] == "1"
        assert d["name"] == "test"
        assert "freshness" in d


class TestMemoryStore:
    def test_remember(self):
        store = MemoryStore(project_dir="/tmp")
        entry = store.remember("test-mem", "A test memory", "This is the body")
        assert entry.name == "test-mem"
        assert entry.id in store._memories

    def test_remember_updates_existing(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("test-mem", "v1", "body v1")
        store.remember("test-mem", "v2", "body v2")
        entry = store.read("test-mem")
        assert entry.description == "v2"
        assert store.stats()["total_memories"] == 1

    def test_forget(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("test-mem", "desc", "body")
        assert store.forget("test-mem") is True
        assert store.read("test-mem") is None

    def test_forget_nonexistent(self):
        store = MemoryStore(project_dir="/tmp")
        assert store.forget("nonexistent") is False

    def test_read_by_id(self):
        store = MemoryStore(project_dir="/tmp")
        entry = store.remember("test-mem", "desc", "body")
        result = store.read(entry.id)
        assert result is not None

    def test_read_by_name(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("test-mem", "desc", "body")
        result = store.read("test-mem")
        assert result is not None

    def test_list_all(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("mem1", "desc1", "body1")
        store.remember("mem2", "desc2", "body2")
        assert len(store.list_all()) == 2

    def test_list_all_filtered_by_type(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("mem1", "desc1", "body1", type=MemoryType.PROJECT)
        store.remember("mem2", "desc2", "body2", type=MemoryType.REFERENCE)
        project_only = store.list_all(type=MemoryType.PROJECT)
        assert len(project_only) == 1

    def test_list_all_filtered_by_scope(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("mem1", "desc1", "body1", scope=FactScope.PROJECT)
        store.remember("mem2", "desc2", "body2", scope=FactScope.GLOBAL)
        project_only = store.list_all(scope=FactScope.PROJECT)
        assert len(project_only) == 1

    def test_search(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("python-tips", "Python tips", "Use list comprehensions for cleaner code")
        store.remember("rust-tips", "Rust tips", "Use ownership patterns for memory safety")
        hits = store.search("python comprehensions")
        assert len(hits) > 0
        assert hits[0].memory.name == "python-tips"

    def test_search_no_results(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("mem1", "desc", "body about python")
        hits = store.search("javascript frameworks")
        assert len(hits) == 0

    def test_search_scores_ranked(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("mem1", "Python Python Python", "Python programming tips")
        store.remember("mem2", "Other topic", "Something else entirely")
        hits = store.search("python")
        assert len(hits) > 0
        assert hits[0].memory.name == "mem1"

    def test_auto_recall(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("mem1", "Python tips", "Use list comprehensions")
        result = store.auto_recall("python programming")
        assert isinstance(result, RecallResult)
        assert result.query == "python programming"

    def test_auto_recall_empty(self):
        store = MemoryStore(project_dir="/tmp")
        result = store.auto_recall("nonexistent topic")
        assert len(result.hits) == 0

    def test_find_duplicates(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("python-tips", "Python programming tips", "Use list comprehensions")
        dups = store.find_duplicates("python-tips", "Python programming tips")
        assert len(dups) > 0

    def test_stats(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("mem1", "desc", "body")
        stats = store.stats()
        assert stats["total_memories"] == 1
        assert "remembered" in stats


class TestAssessRememberWrite:
    def test_auto_allow_project(self):
        store = MemoryStore(project_dir="/tmp")
        assessment = assess_remember_write(
            store, "test-mem", "A description", "A body",
            type=MemoryType.PROJECT, scope=FactScope.PROJECT,
        )
        assert assessment["auto_allow"] is True

    def test_block_empty_description(self):
        store = MemoryStore(project_dir="/tmp")
        assessment = assess_remember_write(store, "test", "", "body")
        assert assessment["auto_allow"] is False

    def test_block_global_scope(self):
        store = MemoryStore(project_dir="/tmp")
        assessment = assess_remember_write(
            store, "test", "desc", "body", scope=FactScope.GLOBAL,
        )
        assert assessment["auto_allow"] is False

    def test_block_feedback_type(self):
        store = MemoryStore(project_dir="/tmp")
        assessment = assess_remember_write(
            store, "test", "desc", "body", type=MemoryType.FEEDBACK,
        )
        assert assessment["auto_allow"] is False

    def test_block_too_long(self):
        store = MemoryStore(project_dir="/tmp")
        assessment = assess_remember_write(
            store, "test", "desc", "x" * 7000,
        )
        assert assessment["auto_allow"] is False

    def test_block_duplicate(self):
        store = MemoryStore(project_dir="/tmp")
        store.remember("python-tips", "Python programming tips", "Use list comprehensions for Python")
        assessment = assess_remember_write(
            store, "python-tips", "Python programming tips", "Use list comprehensions for Python",
        )
        assert assessment["auto_allow"] is False
        assert "duplicate" in assessment["reason"].lower()
