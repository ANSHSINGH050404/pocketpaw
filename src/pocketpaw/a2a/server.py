# A2A Protocol — Server implementation for PocketPaw.
#
# Phase 1: Exposes PocketPaw as an A2A-compatible agent.
#
# Endpoints:
#   GET  /.well-known/agent.json  — Agent Card (capability manifest)
#   POST /a2a/tasks/send          — Submit a task (returns task JSON)
#   POST /a2a/tasks/send/stream   — Submit a task; SSE response stream
#   GET  /a2a/tasks/{task_id}     — Poll task status
#   POST /a2a/tasks/{task_id}/cancel — Cancel an in-flight task
#
# All task processing is routed through the existing PocketPaw AgentLoop
# via the internal message bus (same path as REST /api/v1/chat).

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from importlib.metadata import version as _pkg_version
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from pocketpaw.a2a.models import (
    A2AMessage,
    AgentCapabilities,
    AgentCard,
    Task,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TextPart,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory task store (sufficient for single-process; Phase 3 may persist)
# ---------------------------------------------------------------------------
_tasks: dict[str, Task] = {}
_cancel_events: dict[str, asyncio.Event] = {}  # task_id → cancellation flag


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# The agent-card route lives at the well-known path (outside /api prefix)
well_known_router = APIRouter(tags=["A2A"])

# Task endpoints live under /a2a
tasks_router = APIRouter(prefix="/a2a/tasks", tags=["A2A"])


# ---------------------------------------------------------------------------
# Agent Card helpers
# ---------------------------------------------------------------------------


def _build_agent_card(base_url: str) -> AgentCard:
    """Build an A2A-compliant Agent Card from current PocketPaw config."""
    # Advertise a skill per enabled tool profile group
    skills: list[dict[str, Any]] = [
        {
            "id": "general-assistant",
            "name": "General Assistant",
            "description": (
                "Answer questions, run shell commands, search the web, manage files, "
                "read/send email, control Spotify, and more."
            ),
            "input_modes": ["text"],
            "output_modes": ["text"],
        }
    ]

    try:
        paw_version = _pkg_version("pocketpaw")
    except Exception:
        paw_version = "unknown"

    return AgentCard(
        name="PocketPaw",
        description=(
            "Self-hosted, modular AI agent. "
            "Runs locally with 60+ built-in tools across productivity, coding, and research."
        ),
        url=base_url,
        version=paw_version,
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True,
        ),
        skills=skills,
    )


# ---------------------------------------------------------------------------
# Internal: bridge A2A task to AgentLoop via message bus
# ---------------------------------------------------------------------------


class _A2ASessionBridge:
    """Bridges the message bus outbound stream to an asyncio Queue for A2A SSE.

    Mirrors the pattern in ``api/v1/chat.py::_APISessionBridge``.
    """

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self._outbound_cb = None
        self._system_cb = None

    async def start(self) -> None:
        from pocketpaw.bus import get_message_bus
        from pocketpaw.bus.events import Channel, OutboundMessage, SystemEvent

        bus = get_message_bus()

        async def _on_outbound(msg: OutboundMessage) -> None:
            if msg.chat_id != self.chat_id:
                return
            if msg.is_stream_chunk:
                await self.queue.put({"type": "chunk", "content": msg.content})
            elif msg.is_stream_end:
                await self.queue.put({"type": "stream_end"})
            else:
                await self.queue.put({"type": "chunk", "content": msg.content})

        async def _on_system(evt: SystemEvent) -> None:
            data = evt.data or {}
            sk = data.get("session_key", "")
            if not sk or not sk.endswith(f":{self.chat_id}"):
                return
            if evt.event_type == "error":
                await self.queue.put({"type": "error", "message": data.get("message", "")})

        self._outbound_cb = _on_outbound
        self._system_cb = _on_system
        bus.subscribe_outbound(Channel.WEBSOCKET, _on_outbound)
        bus.subscribe_system(_on_system)

    async def stop(self) -> None:
        from pocketpaw.bus import get_message_bus
        from pocketpaw.bus.events import Channel

        bus = get_message_bus()
        if self._outbound_cb:
            bus.unsubscribe_outbound(Channel.WEBSOCKET, self._outbound_cb)
        if self._system_cb:
            bus.unsubscribe_system(self._system_cb)


async def _dispatch_to_agent(task_id: str, message: A2AMessage) -> str:
    """Publish an inbound message onto the bus and return the chat_id."""
    from pocketpaw.bus import get_message_bus
    from pocketpaw.bus.events import Channel, InboundMessage

    # Use task_id as the stable chat_id so session context is maintained
    chat_id = f"a2a:{task_id}"
    text = " ".join(part.text for part in message.parts if part.type == "text")

    msg = InboundMessage(
        channel=Channel.WEBSOCKET,
        sender_id="a2a_client",
        chat_id=chat_id,
        content=text,
        metadata={"source": "a2a_protocol", "task_id": task_id},
    )
    bus = get_message_bus()
    await bus.publish_inbound(msg)
    return chat_id


# ---------------------------------------------------------------------------
# Endpoint: Agent Card
# ---------------------------------------------------------------------------


@well_known_router.get("/.well-known/agent.json", response_class=JSONResponse)
async def get_agent_card(request: Request):
    """Return the A2A Agent Card describing PocketPaw's capabilities.

    External agents discover this endpoint first when initiating A2A sessions.
    """
    base_url = str(request.base_url).rstrip("/")
    card = _build_agent_card(base_url)
    return JSONResponse(content=card.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Endpoint: tasks/send (non-streaming)
# ---------------------------------------------------------------------------


@tasks_router.post("/send", response_model=Task)
async def tasks_send(params: TaskSendParams):
    """Submit a task to PocketPaw and wait for the completed response.

    The task is routed through the internal AgentLoop. Completes when the
    agent signals ``stream_end`` or on a 120-second timeout.
    """
    task_id = params.id  # always set: default_factory in TaskSendParams guarantees this
    chat_id = f"a2a:{task_id}"

    # Create cancel event so tasks_cancel can interrupt even non-streaming tasks
    cancel_event = asyncio.Event()
    _cancel_events[task_id] = cancel_event

    # Record initial task state
    task = Task(
        id=task_id,
        session_id=params.session_id,
        status=TaskStatus(state=TaskState.SUBMITTED),
        history=[params.message],
        metadata=params.metadata,
    )
    _tasks[task_id] = task

    bridge = _A2ASessionBridge(chat_id)
    await bridge.start()

    # Transition: submitted → working
    task.status = TaskStatus(state=TaskState.WORKING)

    try:
        await _dispatch_to_agent(task_id, params.message)

        # Collect chunks until stream_end, error, or cancellation.
        # Use asyncio.wait so a cancel_event.wait() future can race against
        # bridge.queue.get() — this makes cancellation work for sync tasks too.
        cancel_fut = asyncio.ensure_future(cancel_event.wait())
        content_parts: list[str] = []
        done = False
        while not done:
            get_fut = asyncio.ensure_future(bridge.queue.get())
            try:
                finished, _ = await asyncio.wait(
                    {get_fut, cancel_fut},
                    timeout=120,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except Exception:
                finished = set()

            if not finished:  # timeout
                get_fut.cancel()
                cancel_fut.cancel()
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message=A2AMessage(
                        role="agent",
                        parts=[TextPart(text="Task timed out after 120 seconds.")],
                    ),
                )
                return task

            if cancel_fut in finished:  # cancellation requested
                get_fut.cancel()
                task.status = TaskStatus(state=TaskState.CANCELED)
                return task

            # get_fut completed — process the event
            event = get_fut.result()
            if event["type"] == "chunk":
                content_parts.append(event.get("content", ""))
            elif event["type"] == "stream_end":
                done = True
                cancel_fut.cancel()
            elif event["type"] == "error":
                cancel_fut.cancel()
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message=A2AMessage(
                        role="agent",
                        parts=[TextPart(text=event.get("message", "Agent error"))],
                    ),
                )
                return task

        # Build completion response
        agent_reply = A2AMessage(
            role="agent",
            parts=[TextPart(text="".join(content_parts))],
        )
        task.status = TaskStatus(state=TaskState.COMPLETED, message=agent_reply)
        task.history.append(agent_reply)
        return task

    finally:
        await bridge.stop()


# ---------------------------------------------------------------------------
# Endpoint: tasks/send/stream (SSE streaming)
# ---------------------------------------------------------------------------


@tasks_router.post("/send/stream")
async def tasks_send_stream(params: TaskSendParams):
    """Submit a task and receive an SSE stream of state-update events.

    Each SSE event carries a JSON-encoded ``TaskStatus`` update:
    - ``submitted``  — initial acknowledgment
    - ``working``    — agent began processing, with delta content parts
    - ``completed``  — final message and task result
    - ``failed``     — unrecoverable error
    """
    task_id = params.id  # always set: default_factory in TaskSendParams guarantees this
    chat_id = f"a2a:{task_id}"

    cancel_event = asyncio.Event()
    _cancel_events[task_id] = cancel_event

    task = Task(
        id=task_id,
        session_id=params.session_id,
        status=TaskStatus(state=TaskState.SUBMITTED),
        history=[params.message],
        metadata=params.metadata,
    )
    _tasks[task_id] = task

    bridge = _A2ASessionBridge(chat_id)
    await bridge.start()
    await _dispatch_to_agent(task_id, params.message)

    async def _event_generator():
        try:
            # Submitted acknowledgment
            submitted_status = TaskStatus(state=TaskState.SUBMITTED)
            yield _format_sse(
                "task_status_update",
                {
                    "id": task_id,
                    "status": submitted_status.model_dump(mode="json"),
                    "final": False,
                },
            )

            # Working notification
            working_status = TaskStatus(state=TaskState.WORKING)
            _tasks[task_id].status = working_status
            yield _format_sse(
                "task_status_update",
                {
                    "id": task_id,
                    "status": working_status.model_dump(mode="json"),
                    "final": False,
                },
            )

            # Stream content chunks as working updates
            accumulated: list[str] = []
            while not cancel_event.is_set():
                try:
                    event = await asyncio.wait_for(bridge.queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                if event["type"] == "chunk":
                    chunk_text = event.get("content", "")
                    accumulated.append(chunk_text)
                    delta_msg = A2AMessage(role="agent", parts=[TextPart(text=chunk_text)])
                    chunk_status = TaskStatus(state=TaskState.WORKING, message=delta_msg)
                    yield _format_sse(
                        "task_status_update",
                        {
                            "id": task_id,
                            "status": chunk_status.model_dump(mode="json"),
                            "final": False,
                        },
                    )

                elif event["type"] == "stream_end":
                    full_text = "".join(accumulated)
                    agent_reply = A2AMessage(role="agent", parts=[TextPart(text=full_text)])
                    completed_status = TaskStatus(state=TaskState.COMPLETED, message=agent_reply)
                    _tasks[task_id].status = completed_status
                    _tasks[task_id].history.append(agent_reply)
                    yield _format_sse(
                        "task_status_update",
                        {
                            "id": task_id,
                            "status": completed_status.model_dump(mode="json"),
                            "final": True,
                        },
                    )
                    break

                elif event["type"] == "error":
                    agent_reply = A2AMessage(
                        role="agent",
                        parts=[TextPart(text=event.get("message", "Agent error"))],
                    )
                    failed_status = TaskStatus(state=TaskState.FAILED, message=agent_reply)
                    _tasks[task_id].status = failed_status
                    yield _format_sse(
                        "task_status_update",
                        {
                            "id": task_id,
                            "status": failed_status.model_dump(mode="json"),
                            "final": True,
                        },
                    )
                    break

        finally:
            await bridge.stop()
            _cancel_events.pop(task_id, None)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Endpoint: tasks/{task_id} (GET — poll)
# ---------------------------------------------------------------------------


@tasks_router.get("/{task_id}", response_model=Task)
async def tasks_get(task_id: str):
    """Poll the current status of a previously submitted task."""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


# ---------------------------------------------------------------------------
# Endpoint: tasks/{task_id}/cancel
# ---------------------------------------------------------------------------


@tasks_router.post("/{task_id}/cancel")
async def tasks_cancel(task_id: str):
    """Request cancellation of an in-flight task."""
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    cancel_event = _cancel_events.get(task_id)
    if cancel_event:
        cancel_event.set()

    task.status = TaskStatus(
        state=TaskState.CANCELED,
        timestamp=datetime.now(tz=UTC),
    )
    return {"id": task_id, "status": "canceled"}


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _format_sse(event: str, data: dict) -> str:
    """Format a single SSE message frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# Router registration helper
# ---------------------------------------------------------------------------


def register_routes(app) -> None:
    """Mount all A2A routers onto *app*.

    Called during dashboard startup. Keeps dashboard.py free of A2A-specific
    import details and makes it easy for maintainers to see what gets exposed.

    Routes added:
      GET  /.well-known/agent.json     — Agent Card capability manifest
      POST /a2a/tasks/send             — Submit task (blocking)
      POST /a2a/tasks/send/stream      — Submit task (SSE streaming)
      GET  /a2a/tasks/{task_id}        — Poll task status
      POST /a2a/tasks/{task_id}/cancel — Cancel task
    """
    app.include_router(well_known_router)
    app.include_router(tasks_router)
