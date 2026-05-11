from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request

from agent_runtime.callbacks import CallbackClient
from agent_runtime.graph import run_task_with_langgraph
from agent_runtime.i18n import Localizer, locale_from_metadata, normalize_locale
from agent_runtime.llm import AgentLLM, build_agent_llm_from_runtime_config
from agent_runtime.schemas import Task, TaskEvent
from agent_runtime.tools import NullSearchTool, SearchTool


active_tasks: dict[str, asyncio.Task[None]] = {}
active_task_payloads: dict[str, Task] = {}


def verify_worker_token(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("AGENT_INTERNAL_TOKEN", "").strip()
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid internal token")


def get_callbacks() -> CallbackClient:
    return CallbackClient()


def get_search_tool() -> SearchTool:
    return NullSearchTool()


def get_agent_llm(request: Request) -> AgentLLM:
    return _ensure_agent_llm(request.app)


def _ensure_agent_llm(app: FastAPI) -> AgentLLM:
    llm = getattr(app.state, "agent_llm", None)
    if llm is None:
        llm = build_agent_llm_from_runtime_config(getattr(app.state, "runtime_config", None))
        app.state.agent_llm = llm
    return llm


def _localizer_for_task(task: Task | None = None) -> Localizer:
    if task and task.locale:
        return Localizer(normalize_locale(task.locale))
    if task:
        return Localizer(locale_from_metadata(task.metadata))
    return Localizer()


def create_app(runtime_config: dict[str, Any] | None = None, llm: AgentLLM | None = None) -> FastAPI:
    app = FastAPI(title="CyberVerse Agent Worker")
    app.state.runtime_config = runtime_config or {}
    app.state.agent_llm = llm

    @app.get("/v1/health")
    async def health() -> dict[str, object]:
        llm = _ensure_agent_llm(app)
        return {
            "status": "ok",
            "active_tasks": len(active_tasks),
            "llm_provider": getattr(llm, "provider", ""),
            "llm_model": getattr(llm, "model", ""),
        }

    @app.post("/v1/tasks/{task_id}/run", status_code=202, dependencies=[Depends(verify_worker_token)])
    async def run_task(
        task_id: str,
        task: Task,
        background_tasks: BackgroundTasks,
        callbacks: CallbackClient = Depends(get_callbacks),
        search_tool: SearchTool = Depends(get_search_tool),
        llm: AgentLLM = Depends(get_agent_llm),
    ) -> dict[str, str]:
        if task.id != task_id:
            raise HTTPException(status_code=400, detail="task id mismatch")
        if task_id in active_tasks and not active_tasks[task_id].done():
            return {"status": "already_running", "task_id": task_id}

        background_tasks.add_task(_start_task, task, callbacks, search_tool, llm)
        return {"status": "accepted", "task_id": task_id}

    @app.post("/v1/tasks/{task_id}/cancel", dependencies=[Depends(verify_worker_token)])
    async def cancel_task(task_id: str, callbacks: CallbackClient = Depends(get_callbacks)) -> dict[str, str]:
        task = active_tasks.get(task_id)
        if task and not task.done():
            task.cancel()
            task_payload = active_task_payloads.get(task_id)
            await callbacks.event(
                task_id,
                TaskEvent(
                    event_type="task.cancelled",
                    status="cancelled",
                    message=_localizer_for_task(task_payload).text("worker.cancelled"),
                    progress=0,
                ),
            )
        return {"status": "cancelled", "task_id": task_id}

    return app


async def _start_task(task: Task, callbacks: CallbackClient, search_tool: SearchTool, llm: AgentLLM) -> None:
    runner = asyncio.create_task(_run_task(task, callbacks, search_tool, llm))
    active_tasks[task.id] = runner
    active_task_payloads[task.id] = task
    try:
        await runner
    finally:
        active_tasks.pop(task.id, None)
        active_task_payloads.pop(task.id, None)


async def _run_task(task: Task, callbacks: CallbackClient, search_tool: SearchTool, llm: AgentLLM) -> None:
    try:
        await run_task_with_langgraph(task, search_tool, callbacks, llm=llm)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        localizer = _localizer_for_task(task)
        await callbacks.event(
            task.id,
            TaskEvent(
                event_type="task.failed",
                status="failed",
                message=localizer.text("worker.failed", error=str(exc)),
                progress=task.progress,
            ),
        )
        raise


app = create_app()
