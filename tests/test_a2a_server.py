# Tests for A2A Protocol — Phase 1: Server implementation.
# Created: 2026-03-07
#
# Covers:
#  - Agent Card structure and required fields
#  - POST /a2a/tasks/send (non-streaming, mocked AgentLoop)
#  - GET  /a2a/tasks/{task_id} polling
#  - POST /a2a/tasks/{task_id}/cancel
#  - POST /a2a/tasks/send/stream (SSE format validation)
#  - Model validation: TaskState enum, TextPart, A2AMessage

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pocketpaw.a2a.models import (
    A2AMessage,
    AgentCard,
    Task,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
)
from pocketpaw.a2a.server import (
    _A2ASessionBridge,
    _cancel_events,
    _format_sse,
    _tasks,
    tasks_router,
    well_known_router,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_app():
    """Minimal FastAPI app with both A2A routers mounted."""
    app = FastAPI()
    app.include_router(well_known_router)
    app.include_router(tasks_router)
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clear_task_store():
    """Isolate the in-memory task store between tests."""
    _tasks.clear()
    _cancel_events.clear()
    yield
    _tasks.clear()
    _cancel_events.clear()


def _make_send_params(text: str = "Hello PocketPaw", task_id: str = "test-task-001"):
    return TaskSendParams(
        id=task_id,
        message=A2AMessage(role="user", parts=[TextPart(text=text)]),
    )


# ---------------------------------------------------------------------------
# Tests: Pydantic Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_task_state_values(self):
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.WORKING == "working"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.FAILED == "failed"
        assert TaskState.CANCELED == "canceled"

    def test_text_part_defaults(self):
        part = TextPart(text="hello")
        assert part.type == "text"
        assert part.text == "hello"

    def test_a2a_message_serialization(self):
        msg = A2AMessage(role="agent", parts=[TextPart(text="hi")])
        data = msg.model_dump()
        assert data["role"] == "agent"
        assert data["parts"][0]["text"] == "hi"

    def test_agent_card_defaults(self):
        card = AgentCard(
            name="Test", description="Desc", url="http://localhost:8888", version="1.0"
        )
        assert card.capabilities.streaming is True
        assert card.default_input_modes == ["text"]
        assert card.default_output_modes == ["text"]

    def test_task_send_params_auto_id(self):
        params = TaskSendParams(message=A2AMessage(role="user", parts=[TextPart(text="test")]))
        assert params.id  # auto-generated

    def test_task_status_default_timestamp(self):
        status = TaskStatus(state=TaskState.SUBMITTED)
        assert status.timestamp is not None


# ---------------------------------------------------------------------------
# Tests: GET /.well-known/agent.json
# ---------------------------------------------------------------------------


class TestAgentCard:
    def test_agent_card_returns_200(self, client):
        resp = client.get("/.well-known/agent.json")
        assert resp.status_code == 200

    def test_agent_card_content_type(self, client):
        resp = client.get("/.well-known/agent.json")
        assert "application/json" in resp.headers.get("content-type", "")

    def test_agent_card_required_fields(self, client):
        resp = client.get("/.well-known/agent.json")
        data = resp.json()
        assert "name" in data
        assert "description" in data
        assert "url" in data
        assert "version" in data
        assert "capabilities" in data
        assert "skills" in data

    def test_agent_card_name(self, client):
        resp = client.get("/.well-known/agent.json")
        assert resp.json()["name"] == "PocketPaw"

    def test_agent_card_capabilities_streaming(self, client):
        resp = client.get("/.well-known/agent.json")
        caps = resp.json()["capabilities"]
        assert caps["streaming"] is True

    def test_agent_card_skills_list(self, client):
        resp = client.get("/.well-known/agent.json")
        skills = resp.json()["skills"]
        assert isinstance(skills, list)
        assert len(skills) >= 1
        skill = skills[0]
        assert "id" in skill
        assert "name" in skill
        assert "description" in skill


# ---------------------------------------------------------------------------
# Tests: POST /a2a/tasks/send (non-streaming)
# ---------------------------------------------------------------------------


class TestTasksSend:
    @patch("pocketpaw.a2a.server._dispatch_to_agent")
    @patch("pocketpaw.a2a.server._A2ASessionBridge")
    def test_send_returns_completed_task(self, mock_bridge_cls, mock_dispatch, client):
        bridge = MagicMock()
        q = asyncio.Queue()
        bridge.queue = q
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        mock_bridge_cls.return_value = bridge
        mock_dispatch.return_value = "a2a:test-task-001"

        async def _load():
            await q.put({"type": "chunk", "content": "Here is the answer."})
            await q.put({"type": "stream_end"})

        asyncio.get_event_loop().run_until_complete(_load())

        params = _make_send_params()
        resp = client.post("/a2a/tasks/send", content=params.model_dump_json())
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "test-task-001"
        assert data["status"]["state"] == TaskState.COMPLETED
        assert "Here is the answer." in data["status"]["message"]["parts"][0]["text"]

    @patch("pocketpaw.a2a.server._dispatch_to_agent")
    @patch("pocketpaw.a2a.server._A2ASessionBridge")
    def test_send_failed_on_error_event(self, mock_bridge_cls, mock_dispatch, client):
        bridge = MagicMock()
        q = asyncio.Queue()
        bridge.queue = q
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        mock_bridge_cls.return_value = bridge
        mock_dispatch.return_value = "a2a:test-task-001"

        async def _load():
            await q.put({"type": "error", "message": "Something went wrong"})

        asyncio.get_event_loop().run_until_complete(_load())

        params = _make_send_params()
        resp = client.post("/a2a/tasks/send", content=params.model_dump_json())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]["state"] == TaskState.FAILED

    def test_send_invalid_body_returns_422(self, client):
        resp = client.post("/a2a/tasks/send", json={"bad": "body"})
        assert resp.status_code == 422

    @patch("pocketpaw.a2a.server._dispatch_to_agent")
    @patch("pocketpaw.a2a.server._A2ASessionBridge")
    def test_send_stores_task_history(self, mock_bridge_cls, mock_dispatch, client):
        bridge = MagicMock()
        q = asyncio.Queue()
        bridge.queue = q
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        mock_bridge_cls.return_value = bridge
        mock_dispatch.return_value = "a2a:test-task-001"

        async def _load():
            await q.put({"type": "chunk", "content": "Done."})
            await q.put({"type": "stream_end"})

        asyncio.get_event_loop().run_until_complete(_load())

        params = _make_send_params()
        resp = client.post("/a2a/tasks/send", content=params.model_dump_json())
        data = resp.json()
        # history should include the original user message + agent reply
        assert len(data["history"]) >= 2
        assert data["history"][0]["role"] == "user"
        assert data["history"][-1]["role"] == "agent"


# ---------------------------------------------------------------------------
# Tests: GET /a2a/tasks/{task_id}
# ---------------------------------------------------------------------------


class TestTasksGet:
    def test_get_task_not_found(self, client):
        resp = client.get("/a2a/tasks/nonexistent-id")
        assert resp.status_code == 404

    def test_get_task_found(self, client):
        # Pre-seed the store
        task = Task(
            id="known-task",
            status=TaskStatus(state=TaskState.WORKING),
        )
        _tasks["known-task"] = task

        resp = client.get("/a2a/tasks/known-task")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "known-task"
        assert data["status"]["state"] == TaskState.WORKING


# ---------------------------------------------------------------------------
# Tests: POST /a2a/tasks/{task_id}/cancel
# ---------------------------------------------------------------------------


class TestTasksCancel:
    def test_cancel_nonexistent_task(self, client):
        resp = client.post("/a2a/tasks/ghost-task/cancel")
        assert resp.status_code == 404

    def test_cancel_sets_state(self, client):
        task = Task(
            id="cancel-me",
            status=TaskStatus(state=TaskState.WORKING),
        )
        _tasks["cancel-me"] = task
        cancel_evt = asyncio.Event()
        _cancel_events["cancel-me"] = cancel_evt

        resp = client.post("/a2a/tasks/cancel-me/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "canceled"
        assert _tasks["cancel-me"].status.state == TaskState.CANCELED
        assert cancel_evt.is_set()


# ---------------------------------------------------------------------------
# Tests: POST /a2a/tasks/send/stream (SSE)
# ---------------------------------------------------------------------------


class TestTasksSendStream:
    @patch("pocketpaw.a2a.server._dispatch_to_agent")
    @patch("pocketpaw.a2a.server._A2ASessionBridge")
    def test_stream_returns_sse_content_type(self, mock_bridge_cls, mock_dispatch, client):
        bridge = MagicMock()
        q = asyncio.Queue()
        bridge.queue = q
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        mock_bridge_cls.return_value = bridge
        mock_dispatch.return_value = "a2a:stream-task"

        async def _load():
            await q.put({"type": "chunk", "content": "Streaming..."})
            await q.put({"type": "stream_end"})

        asyncio.get_event_loop().run_until_complete(_load())
        params = _make_send_params(task_id="stream-task")
        with client.stream(
            "POST", "/a2a/tasks/send/stream", content=params.model_dump_json()
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

    @patch("pocketpaw.a2a.server._dispatch_to_agent")
    @patch("pocketpaw.a2a.server._A2ASessionBridge")
    def test_stream_sse_events_valid_format(self, mock_bridge_cls, mock_dispatch, client):
        """Each SSE event must follow: event: <type>\\ndata: <json>\\n\\n"""
        bridge = MagicMock()
        q = asyncio.Queue()
        bridge.queue = q
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        mock_bridge_cls.return_value = bridge
        mock_dispatch.return_value = "a2a:sse-val"

        async def _load():
            await q.put({"type": "chunk", "content": "partial answer"})
            await q.put({"type": "stream_end"})

        asyncio.get_event_loop().run_until_complete(_load())
        params = _make_send_params(task_id="sse-val")
        with client.stream(
            "POST", "/a2a/tasks/send/stream", content=params.model_dump_json()
        ) as resp:
            raw = resp.read().decode()

        events = [e.strip() for e in raw.split("\n\n") if e.strip()]
        assert len(events) >= 2

        for block in events:
            lines = block.split("\n")
            event_line = next((line for line in lines if line.startswith("event:")), None)
            data_line = next((line for line in lines if line.startswith("data:")), None)
            assert event_line is not None, f"Missing 'event:' in: {block!r}"
            assert data_line is not None, f"Missing 'data:' in: {block!r}"
            data_str = data_line.split(":", 1)[1].strip()
            parsed = json.loads(data_str)
            assert "id" in parsed
            assert "status" in parsed

    @patch("pocketpaw.a2a.server._dispatch_to_agent")
    @patch("pocketpaw.a2a.server._A2ASessionBridge")
    def test_stream_final_event_has_completed_state(self, mock_bridge_cls, mock_dispatch, client):
        bridge = MagicMock()
        q = asyncio.Queue()
        bridge.queue = q
        bridge.start = AsyncMock()
        bridge.stop = AsyncMock()
        mock_bridge_cls.return_value = bridge
        mock_dispatch.return_value = "a2a:final-test"

        async def _load():
            await q.put({"type": "chunk", "content": "Final answer."})
            await q.put({"type": "stream_end"})

        asyncio.get_event_loop().run_until_complete(_load())
        params = _make_send_params(task_id="final-test")
        with client.stream(
            "POST", "/a2a/tasks/send/stream", content=params.model_dump_json()
        ) as resp:
            raw = resp.read().decode()

        events = [e.strip() for e in raw.split("\n\n") if e.strip()]
        final = events[-1]
        data_line = next(line for line in final.split("\n") if line.startswith("data:"))
        parsed = json.loads(data_line.split(":", 1)[1].strip())
        assert parsed["final"] is True
        assert parsed["status"]["state"] == TaskState.COMPLETED


# ---------------------------------------------------------------------------
# Tests: SSE format helper
# ---------------------------------------------------------------------------


class TestFormatSSE:
    def test_format_sse_basic(self):
        result = _format_sse("test_event", {"key": "value"})
        assert result.startswith("event: test_event\n")
        assert "data:" in result
        assert result.endswith("\n\n")

    def test_format_sse_json_valid(self):
        result = _format_sse("update", {"id": "t1", "status": "working"})
        data_line = [line for line in result.split("\n") if line.startswith("data:")][0]
        parsed = json.loads(data_line.split(":", 1)[1].strip())
        assert parsed["id"] == "t1"


# ---------------------------------------------------------------------------
# Tests: _A2ASessionBridge
# ---------------------------------------------------------------------------


class TestA2ASessionBridge:
    @pytest.mark.asyncio
    async def test_bridge_creation(self):
        bridge = _A2ASessionBridge("chat-abc")
        assert bridge.chat_id == "chat-abc"
        assert isinstance(bridge.queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_bridge_queue_chunk(self):
        bridge = _A2ASessionBridge("x")
        await bridge.queue.put({"type": "chunk", "content": "hello"})
        item = await bridge.queue.get()
        assert item["type"] == "chunk"
        assert item["content"] == "hello"
