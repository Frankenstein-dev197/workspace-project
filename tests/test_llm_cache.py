"""Tests for LLM cache with drift detection."""

import pytest

from daemon_engine.models.llm_cache import (
    LLMCache,
    ApiKind,
    StructuralHash,
    DriftReport,
    extract_cache_hot_zone,
    compute_drift,
    canonicalize_for_hash,
    sha256_hex,
    conversation_discriminator,
)


class TestCanonicalization:
    def test_canonicalize_dict(self):
        result = canonicalize_for_hash({"b": 2, "a": 1})
        assert list(result.keys()) == ["a", "b"]

    def test_canonicalize_nested(self):
        result = canonicalize_for_hash({"z": {"y": 2, "x": 1}})
        assert list(result["z"].keys()) == ["x", "y"]

    def test_canonicalize_list(self):
        result = canonicalize_for_hash([3, 1, 2])
        assert result == [3, 1, 2]

    def test_sha256_hex_stable(self):
        h1 = sha256_hex({"a": 1, "b": 2})
        h2 = sha256_hex({"b": 2, "a": 1})
        assert h1 == h2

    def test_sha256_hex_different(self):
        h1 = sha256_hex({"a": 1})
        h2 = sha256_hex({"a": 2})
        assert h1 != h2


class TestHotZone:
    def test_extract_hot_zone(self):
        messages = [{"role": "user", "content": "Hello"}]
        hot = extract_cache_hot_zone(messages, system_prompt="You are an agent")
        assert hot.system != ""
        assert len(hot.early_messages) == 1

    def test_extract_with_tools(self):
        tools = [{"name": "bash"}, {"name": "read"}]
        hot = extract_cache_hot_zone([], tools=tools)
        assert hot.tools != ""

    def test_extract_empty(self):
        hot = extract_cache_hot_zone([])
        assert hot.system == ""
        assert hot.tools == ""
        assert hot.early_messages == []

    def test_early_message_window(self):
        messages = [{"content": f"msg{i}"} for i in range(10)]
        hot = extract_cache_hot_zone(messages)
        assert len(hot.early_messages) == 3


class TestDriftDetection:
    def test_no_drift(self):
        prev = extract_cache_hot_zone(
            [{"role": "user", "content": "Hello"}],
            system_prompt="System",
        )
        curr = extract_cache_hot_zone(
            [{"role": "user", "content": "Hello"}],
            system_prompt="System",
        )
        drifted = compute_drift(prev, curr)
        assert drifted == []

    def test_system_drift(self):
        prev = extract_cache_hot_zone([], system_prompt="System v1")
        curr = extract_cache_hot_zone([], system_prompt="System v2")
        drifted = compute_drift(prev, curr)
        assert "system" in drifted

    def test_tools_drift(self):
        prev = extract_cache_hot_zone([], tools=[{"name": "bash"}])
        curr = extract_cache_hot_zone([], tools=[{"name": "read"}])
        drifted = compute_drift(prev, curr)
        assert "tools" in drifted

    def test_early_message_drift(self):
        prev = extract_cache_hot_zone([{"content": "msg1"}])
        curr = extract_cache_hot_zone([{"content": "msg1_changed"}])
        drifted = compute_drift(prev, curr)
        assert "early_message[0]" in drifted

    def test_growing_messages_no_drift(self):
        prev = extract_cache_hot_zone([{"content": "msg1"}])
        curr = extract_cache_hot_zone([{"content": "msg1"}, {"content": "msg2"}])
        drifted = compute_drift(prev, curr)
        assert "early_message[0]" not in drifted


class TestConversationDiscriminator:
    def test_same_first_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert conversation_discriminator(msgs) == conversation_discriminator(msgs)

    def test_different_first_message(self):
        m1 = [{"role": "user", "content": "Hello"}]
        m2 = [{"role": "user", "content": "Goodbye"}]
        assert conversation_discriminator(m1) != conversation_discriminator(m2)

    def test_empty_messages(self):
        assert conversation_discriminator([]) == "empty"


class TestLLMCache:
    def test_put_and_get(self):
        cache = LLMCache()
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "response", system_prompt="System", model="gpt-4")
        result, drift = cache.get(messages, system_prompt="System", model="gpt-4")
        assert result == "response"
        assert drift is not None

    def test_miss(self):
        cache = LLMCache()
        messages = [{"role": "user", "content": "Hello"}]
        result, _ = cache.get(messages, model="gpt-4")
        assert result is None

    def test_hit_increments_counter(self):
        cache = LLMCache()
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "response", model="gpt-4")
        cache.get(messages, model="gpt-4")
        cache.get(messages, model="gpt-4")
        stats = cache.stats()
        assert stats["hits"] == 2

    def test_stats(self):
        cache = LLMCache()
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "response", model="gpt-4")
        cache.get(messages, model="gpt-4")
        cache.get(messages, model="gpt-4", system_prompt="different")
        stats = cache.stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 0
        assert "hit_rate" in stats

    def test_eviction(self):
        cache = LLMCache(max_entries=2)
        for i in range(3):
            msgs = [{"content": f"msg{i}"}]
            cache.put(msgs, f"response{i}", model="gpt-4")
        assert len(cache._cache) == 2

    def test_invalidate(self):
        cache = LLMCache()
        messages = [{"role": "user", "content": "Hello"}]
        cache.put(messages, "response", model="gpt-4")
        assert cache.invalidate(messages, model="gpt-4") is True
        result, _ = cache.get(messages, model="gpt-4")
        assert result is None

    def test_clear(self):
        cache = LLMCache()
        messages = [{"content": "Hello"}]
        cache.put(messages, "response", model="gpt-4")
        cache.clear()
        assert len(cache._cache) == 0

    def test_drift_detection(self):
        cache = LLMCache()
        messages1 = [{"role": "user", "content": "Hello"}]
        cache.get(messages1, system_prompt="System v1", model="gpt-4")
        messages2 = [{"role": "user", "content": "Hello"}]
        _, drift = cache.get(messages2, system_prompt="System v2", model="gpt-4")
        assert drift is not None
        assert drift.is_drift is True
        assert "system" in drift.drifted_dimensions

    def test_first_request(self):
        cache = LLMCache()
        messages = [{"role": "user", "content": "Hello"}]
        _, drift = cache.get(messages, system_prompt="System", model="gpt-4")
        assert drift is not None
        assert drift.is_first_request is True

    def test_check_drift_standalone(self):
        cache = LLMCache()
        messages = [{"content": "Hello"}]
        report1 = cache.check_drift(messages, system_prompt="v1")
        assert report1.is_first_request is True
        report2 = cache.check_drift(messages, system_prompt="v2")
        assert report2.is_drift is True
        assert "system" in report2.drifted_dimensions

    def test_ttl_expiry(self):
        cache = LLMCache(ttl_seconds=0)
        messages = [{"content": "Hello"}]
        cache.put(messages, "response", model="gpt-4")
        import time
        time.sleep(0.1)
        result, _ = cache.get(messages, model="gpt-4")
        assert result is None
