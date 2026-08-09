"""Tests for SSE event stream."""

import json
import threading
import time

import pytest

from daemon_engine.core.sse_events import (
    SSEEvent,
    SSEEventType,
    SSEStream,
    parse_sse_stream,
)


class TestSSEEventType:
    def test_values(self):
        assert SSEEventType.LOG.value == "log"
        assert SSEEventType.PROGRESS.value == "progress"
        assert SSEEventType.RESULT.value == "result"
        assert SSEEventType.ERROR.value == "error"
        assert SSEEventType.STREAM_COMPLETE.value == "stream_complete"

    def test_is_str_enum(self):
        assert SSEEventType.LOG == "log"


class TestSSEEvent:
    def test_creation(self):
        event = SSEEvent(event_type=SSEEventType.LOG, data={"msg": "hello"})
        assert event.event_type == SSEEventType.LOG
        assert event.data == {"msg": "hello"}
        assert event.id  # auto-generated

    def test_to_dict(self):
        event = SSEEvent(event_type=SSEEventType.RESULT, data={"x": 1})
        d = event.to_dict()
        assert d["event_type"] == "result"
        assert d["data"] == {"x": 1}

    def test_to_sse_format(self):
        event = SSEEvent(
            event_type=SSEEventType.LOG,
            data={"message": "test"},
            id="abc123",
        )
        sse = event.to_sse_format()
        assert "id: abc123" in sse
        assert "event: log" in sse
        assert "data:" in sse
        assert sse.endswith("\n\n")

    def test_from_sse_format(self):
        raw = "id: abc123\nevent: log\ndata: {\"message\": \"test\"}\n\n"
        event = SSEEvent.from_sse_format(raw)
        assert event.id == "abc123"
        assert event.event_type == SSEEventType.LOG
        assert event.data == {"message": "test"}

    def test_roundtrip(self):
        original = SSEEvent(
            event_type=SSEEventType.RESULT,
            data={"result": 42, "nested": {"a": 1}},
            id="test-id",
        )
        sse = original.to_sse_format()
        parsed = SSEEvent.from_sse_format(sse)
        assert parsed.id == "test-id"
        assert parsed.event_type == SSEEventType.RESULT
        assert parsed.data == {"result": 42, "nested": {"a": 1}}

    def test_from_sse_format_invalid_json(self):
        raw = "id: abc\nevent: log\ndata: not json\n\n"
        event = SSEEvent.from_sse_format(raw)
        assert event.data == {"raw": "not json"}

    def test_from_sse_format_custom_event_name(self):
        raw = "id: abc\nevent: my_custom_event\ndata: {\"x\": 1}\n\n"
        event = SSEEvent.from_sse_format(raw)
        assert event.event_type == SSEEventType.CUSTOM
        assert event.event_name == "my_custom_event"


class TestSSEStream:
    def test_emit_and_consume(self):
        stream = SSEStream()
        stream.emit(SSEEventType.LOG, {"message": "hello"})
        stream.emit(SSEEventType.RESULT, {"result": 42})
        stream.close()

        events = list(stream.consume(timeout=1.0))
        assert len(events) == 3  # 2 emitted + 1 stream_complete
        assert events[0].event_type == SSEEventType.LOG
        assert events[1].event_type == SSEEventType.RESULT
        assert events[2].event_type == SSEEventType.STREAM_COMPLETE

    def test_emit_after_close_raises(self):
        stream = SSEStream()
        stream.close()
        with pytest.raises(RuntimeError):
            stream.emit(SSEEventType.LOG, {})

    def test_emit_log_helper(self):
        stream = SSEStream()
        event = stream.emit_log("starting", level="info")
        assert event.event_type == SSEEventType.LOG
        assert event.data == {"message": "starting", "level": "info"}
        stream.close()

    def test_emit_progress_helper(self):
        stream = SSEStream()
        event = stream.emit_progress(current=5, total=10, message="halfway")
        assert event.event_type == SSEEventType.PROGRESS
        assert event.data == {"current": 5, "total": 10, "message": "halfway"}
        stream.close()

    def test_emit_result_helper(self):
        stream = SSEStream()
        event = stream.emit_result({"answer": 42})
        assert event.event_type == SSEEventType.RESULT
        assert event.data == {"result": {"answer": 42}}
        stream.close()

    def test_emit_error_helper(self):
        stream = SSEStream()
        event = stream.emit_error("something failed", traceback="trace")
        assert event.event_type == SSEEventType.ERROR
        assert event.data == {"error": "something failed", "traceback": "trace"}
        stream.close()

    def test_event_count(self):
        stream = SSEStream()
        assert stream.event_count == 0
        stream.emit(SSEEventType.LOG, {})
        stream.emit(SSEEventType.LOG, {})
        assert stream.event_count == 2
        stream.close()
        # stream_complete adds 1
        assert stream.event_count == 3

    def test_subscribe(self):
        stream = SSEStream()
        received: list[SSEEvent] = []
        stream.subscribe(lambda e: received.append(e))
        stream.emit(SSEEventType.LOG, {"msg": "hello"})
        stream.close()
        assert len(received) >= 1
        assert received[0].data == {"msg": "hello"}

    def test_unsubscribe(self):
        stream = SSEStream()
        received: list[SSEEvent] = []
        callback = lambda e: received.append(e)
        stream.subscribe(callback)
        stream.emit(SSEEventType.LOG, {"msg": "first"})
        stream.unsubscribe(callback)
        stream.emit(SSEEventType.LOG, {"msg": "second"})
        stream.close()
        # Only first event received
        assert len([r for r in received if r.event_type == SSEEventType.LOG]) == 1

    def test_subscriber_error_doesnt_break_stream(self):
        stream = SSEStream()
        received: list[SSEEvent] = []

        def bad_callback(e):
            raise ValueError("bad")

        def good_callback(e):
            received.append(e)

        stream.subscribe(bad_callback)
        stream.subscribe(good_callback)
        stream.emit(SSEEventType.LOG, {"msg": "hello"})
        stream.close()
        assert len(received) >= 1

    def test_consume_all(self):
        stream = SSEStream()
        stream.emit(SSEEventType.LOG, {"msg": "1"})
        stream.emit(SSEEventType.LOG, {"msg": "2"})
        events = stream.consume_all()
        assert len(events) == 2
        stream.close()

    def test_to_sse_lines(self):
        stream = SSEStream()
        stream.emit(SSEEventType.LOG, {"msg": "hello"})
        stream.close()
        lines = list(stream.to_sse_lines())
        assert len(lines) >= 1
        assert "event: log" in lines[0]

    def test_context_manager(self):
        with SSEStream() as stream:
            stream.emit(SSEEventType.LOG, {"msg": "hello"})
        assert stream.is_closed

    def test_close_idempotent(self):
        stream = SSEStream()
        stream.close()
        stream.close()  # Should not raise
        assert stream.is_closed

    def test_consume_timeout(self):
        stream = SSEStream()
        start = time.time()
        events = list(stream.consume(timeout=0.1))
        elapsed = time.time() - start
        assert len(events) == 0
        assert elapsed < 1.0
        stream.close()


class TestParseSSEStream:
    def test_parse_multiple_events(self):
        raw = (
            "id: 1\nevent: log\ndata: {\"msg\": \"first\"}\n\n"
            "id: 2\nevent: result\ndata: {\"result\": 42}\n\n"
        )
        events = parse_sse_stream(raw)
        assert len(events) == 2
        assert events[0].id == "1"
        assert events[1].id == "2"

    def test_parse_empty(self):
        events = parse_sse_stream("")
        assert events == []

    def test_parse_single_event(self):
        raw = "id: abc\nevent: error\ndata: {\"error\": \"failed\"}\n\n"
        events = parse_sse_stream(raw)
        assert len(events) == 1
        assert events[0].event_type == SSEEventType.ERROR


class TestThreadSafety:
    def test_concurrent_emit(self):
        stream = SSEStream(maxsize=10000)

        def emitter(count: int) -> None:
            for i in range(count):
                stream.emit(SSEEventType.LOG, {"n": i})

        threads = [threading.Thread(target=emitter, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stream.close()
        events = list(stream.consume(timeout=1.0))
        log_events = [e for e in events if e.event_type == SSEEventType.LOG]
        assert len(log_events) == 200

    def test_concurrent_emit_and_consume(self):
        stream = SSEStream(maxsize=10000)
        consumed: list[SSEEvent] = []

        def consumer():
            for event in stream.consume(timeout=2.0):
                consumed.append(event)

        consumer_thread = threading.Thread(target=consumer)
        consumer_thread.start()

        for i in range(100):
            stream.emit(SSEEventType.LOG, {"n": i})

        stream.close()
        consumer_thread.join(timeout=3.0)

        log_events = [e for e in consumed if e.event_type == SSEEventType.LOG]
        assert len(log_events) == 100
