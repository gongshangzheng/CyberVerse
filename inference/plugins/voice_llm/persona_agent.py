from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, TypedDict

from inference.core.registry import import_plugin_class
from inference.core.types import (
    PluginConfig,
    ToolCall,
    ToolDefinition,
    ToolResult,
    VoiceLLMInputEvent,
    VoiceLLMOutputEvent,
    VoiceLLMSessionConfig,
)
from inference.plugins.voice_llm.base import VoiceLLMPlugin

logger = logging.getLogger(__name__)


PERSONA_TOOL_DEFINITIONS = [
    ToolDefinition(
        name="wait_for_more_input",
        description="Use when the user's current utterance is incomplete and more input is needed before responding or acting.",
        parameters={
            "type": "object",
            "properties": {
                "partial_text": {
                    "type": "string",
                    "description": "The partial utterance already heard.",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason why the intent is incomplete.",
                },
            },
            "required": ["partial_text"],
        },
    ),
    ToolDefinition(
        name="create_task",
        description="Create a CyberVerse background research task for search, research, aggregation, or report requests.",
        parameters={
            "type": "object",
            "properties": {
                "user_request": {
                    "type": "string",
                    "description": "The complete normalized user request to run in the background.",
                },
                "title": {
                    "type": "string",
                    "description": "A short human-readable task title.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["research"],
                    "default": "research",
                },
            },
            "required": ["user_request"],
        },
    ),
    ToolDefinition(
        name="get_task_status",
        description="Get the latest active CyberVerse background task status for this session.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="cancel_task",
        description="Cancel the latest active CyberVerse background task for this session.",
        parameters={"type": "object", "properties": {}},
    ),
]

PERSONA_AGENT_INSTRUCTIONS = """你是 CyberVerse 数字人 PersonaAgent，直接通过语音和用户对话。
你需要先判断用户当前表达是否已经构成完整意图。
普通寒暄、问答和闲聊：直接自然回答。
语义明显未完成、像半句话、铺垫、犹豫或还在补充：调用 wait_for_more_input。
搜索、查询热点、调研、整理资料、生成报告或需要较长后台处理：调用 create_task。
询问后台任务进度：调用 get_task_status。
要求取消、停止、不用继续当前后台任务：调用 cancel_task。

"""


class PersonaToolState(TypedDict, total=False):
    call: ToolCall
    session_id: str
    result: dict[str, Any]


@dataclass
class PendingAsyncTask:
    session_id: str
    args: dict[str, Any]
    user_request: str


class PersonaTaskClient:
    def __init__(self, server_url: str, internal_token: str = "") -> None:
        self.server_url = server_url.rstrip("/")
        self.internal_token = internal_token.strip()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.internal_token:
            headers["Authorization"] = f"Bearer {self.internal_token}"
        return headers

    async def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body or {}, ensure_ascii=False).encode("utf-8") if body is not None else None

        def do_request() -> dict[str, Any]:
            req = urllib.request.Request(
                f"{self.server_url}{path}",
                data=data,
                method=method,
                headers=self._headers(),
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        return await asyncio.to_thread(do_request)

    async def create_task(self, session_id: str, args: dict[str, Any]) -> dict[str, Any]:
        user_request = str(args.get("user_request") or args.get("request") or "").strip()
        if not user_request:
            raise ValueError("create_task requires user_request")
        payload = {
            "user_request": user_request,
            "title": str(args.get("title") or "").strip(),
            "kind": str(args.get("kind") or "research").strip() or "research",
        }
        return await self._request("POST", f"/sessions/{session_id}/tasks", payload)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/tasks/{task_id}")

    async def get_task_events(self, task_id: str, after_seq: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        events = await self._request("GET", f"/tasks/{task_id}/events?after_seq={after_seq}&limit={limit}")
        raw_events = events.get("events", [])
        return raw_events if isinstance(raw_events, list) else []

    async def get_task_status(self, session_id: str) -> dict[str, Any]:
        tasks = await self._request("GET", f"/sessions/{session_id}/tasks?limit=10")
        active_statuses = {"queued", "running", "waiting_user"}
        for task in tasks.get("tasks", []):
            if task.get("status") in active_statuses:
                events = await self._request("GET", f"/tasks/{task.get('id')}/events?after_seq=0&limit=20")
                return {"task": task, "events": events.get("events", [])}
        return {"task": None, "events": []}

    async def cancel_task(self, session_id: str) -> dict[str, Any]:
        status = await self.get_task_status(session_id)
        task = status.get("task")
        if not task:
            return {"cancelled": False, "reason": "no_active_task"}
        cancelled = await self._request("POST", f"/tasks/{task.get('id')}/cancel", {})
        return {"cancelled": True, "task": cancelled}


class PersonaAgentPlugin(VoiceLLMPlugin):
    """LangGraph-backed persona wrapper for an underlying realtime omni provider.

    The public gRPC wire shape remains the existing VoiceLLM stream. Native tool
    calls are consumed inside this wrapper and are never forwarded to Go or the UI.
    """

    name = "persona.persona"

    def __init__(self) -> None:
        self.model_provider = "doubao"
        self.model_plugin: VoiceLLMPlugin | None = None
        self.task_client: PersonaTaskClient | None = None
        self.checkpoint_db_path = ""
        self.task_poll_interval_seconds = 1.0
        self.task_monitor_timeout_seconds = 1800.0
        self._tool_graph = None
        self._checkpoint_conn: Any | None = None

    async def initialize(self, config: PluginConfig) -> None:
        self.model_provider = str(config.params.get("model_provider") or "doubao").strip()
        if not self.model_provider or self.model_provider == "persona":
            raise ValueError("persona model_provider must reference a concrete omni provider")

        server_url = str(
            config.params.get("server_url")
            or os.getenv("CYBERVERSE_SERVER_URL")
            or "http://localhost:8080/api/v1"
        )
        internal_token = str(config.params.get("internal_token") or os.getenv("AGENT_INTERNAL_TOKEN") or "")
        self.checkpoint_db_path = str(
            config.params.get("checkpoint_db_path")
            or os.getenv("LANGGRAPH_CHECKPOINT_DB")
            or os.path.join(
                os.getenv("CYBERVERSE_CONFIG_DIR", "."),
                "data",
                "tasks",
                "langgraph_checkpoints.db",
            )
        )
        self.task_poll_interval_seconds = max(
            0.1,
            float(config.params.get("task_poll_interval_seconds") or self.task_poll_interval_seconds),
        )
        self.task_monitor_timeout_seconds = max(
            1.0,
            float(config.params.get("task_monitor_timeout_seconds") or self.task_monitor_timeout_seconds),
        )
        self.task_client = PersonaTaskClient(server_url, internal_token)

        omni_config = config.shared.get("omni", {})
        if not isinstance(omni_config, dict):
            raise ValueError("persona provider requires shared omni config")
        provider_conf = omni_config.get(self.model_provider)
        if not isinstance(provider_conf, dict):
            raise ValueError(f"persona model_provider {self.model_provider!r} is not configured")
        class_path = provider_conf.get("plugin_class")
        if not class_path:
            raise ValueError(f"persona model_provider {self.model_provider!r} has no plugin_class")

        plugin_cls = import_plugin_class(str(class_path))
        model_plugin = plugin_cls()
        params = {k: v for k, v in provider_conf.items() if k != "plugin_class"}
        await model_plugin.initialize(
            PluginConfig(
                plugin_name=f"omni.{self.model_provider}",
                params=params,
                shared=config.shared,
            )
        )
        if not isinstance(model_plugin, VoiceLLMPlugin):
            raise TypeError(f"{class_path} is not a VoiceLLMPlugin")
        self.model_plugin = model_plugin
        self._tool_graph = await self._compile_tool_graph()

    async def shutdown(self) -> None:
        if self.model_plugin is not None:
            await self.model_plugin.shutdown()
        if self._checkpoint_conn is not None:
            await self._checkpoint_conn.close()
            self._checkpoint_conn = None

    async def check_voice(self, session_config: VoiceLLMSessionConfig | None = None) -> None:
        if self.model_plugin is None:
            raise RuntimeError("persona model plugin is not initialized")
        await self.model_plugin.check_voice(session_config)

    async def interrupt(self) -> None:
        if self.model_plugin is not None:
            await self.model_plugin.interrupt()

    async def _compile_tool_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception:
            return None

        graph = StateGraph(PersonaToolState)

        async def execute_tool(state: PersonaToolState) -> PersonaToolState:
            call = state["call"]
            session_id = state["session_id"]
            return {"result": await self._execute_tool_direct(call, session_id)}

        graph.add_node("execute_tool", execute_tool)
        graph.add_edge(START, "execute_tool")
        graph.add_edge("execute_tool", END)
        checkpointer = None
        if self.checkpoint_db_path:
            try:
                import aiosqlite
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                if self.checkpoint_db_path != ":memory:":
                    os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_db_path)), exist_ok=True)
                self._checkpoint_conn = await aiosqlite.connect(self.checkpoint_db_path)
                checkpointer = AsyncSqliteSaver(self._checkpoint_conn)
                await checkpointer.setup()
            except Exception as exc:
                logger.warning("persona checkpoint disabled: %s", exc)
                if self._checkpoint_conn is not None:
                    await self._checkpoint_conn.close()
                    self._checkpoint_conn = None
        if checkpointer is None:
            return graph.compile()
        return graph.compile(checkpointer=checkpointer)

    async def _execute_tool(self, call: ToolCall, session_id: str) -> dict[str, Any]:
        if self._tool_graph is None:
            return await self._execute_tool_direct(call, session_id)
        state = await self._tool_graph.ainvoke(
            {"call": call, "session_id": session_id},
            config={"configurable": {"thread_id": f"{session_id}:{call.id or call.name}"}},
        )
        return dict(state.get("result") or {})

    async def _execute_tool_direct(self, call: ToolCall, session_id: str) -> dict[str, Any]:
        name = call.name.strip()
        args = call.arguments or {}
        if name == "wait_for_more_input":
            return {
                "ok": True,
                "waiting": True,
                "partial_text": str(args.get("partial_text") or "").strip(),
            }

        if self.task_client is None:
            raise RuntimeError("persona task client is not initialized")
        if not session_id:
            raise ValueError("persona tool execution requires session_id")

        if name == "create_task":
            return await self.task_client.create_task(session_id, args)
        if name == "get_task_status":
            return await self.task_client.get_task_status(session_id)
        if name == "cancel_task":
            return await self.task_client.cancel_task(session_id)
        raise ValueError(f"unsupported persona tool: {name}")

    @staticmethod
    def _clean_text(text: Any) -> str:
        return str(text or "").strip()

    @staticmethod
    def _needs_space(left: str, right: str) -> bool:
        if not left or not right:
            return False
        return left[-1].isascii() and right[0].isascii() and left[-1].isalnum() and right[0].isalnum()

    @classmethod
    def _merge_text_segments(cls, segments: list[str]) -> str:
        merged = ""
        for segment in segments:
            text = cls._clean_text(segment)
            if not text:
                continue
            if not merged:
                merged = text
                continue
            if text in merged:
                continue
            if merged in text:
                merged = text
                continue
            separator = " " if cls._needs_space(merged, text) else ""
            merged = f"{merged}{separator}{text}"
        return merged.strip()

    @classmethod
    def _partial_text_for_wait(cls, call: ToolCall, turn_transcripts: list[str]) -> str:
        args = call.arguments or {}
        return cls._clean_text(args.get("partial_text") or args.get("text")) or cls._merge_text_segments(turn_transcripts)

    @classmethod
    def _final_user_text(
        cls,
        call: ToolCall,
        pending_partials: list[str],
        turn_transcripts: list[str],
    ) -> str:
        args = call.arguments or {}
        tool_text = cls._clean_text(args.get("user_request") or args.get("request") or args.get("text"))
        current_text = tool_text or cls._merge_text_segments(turn_transcripts)
        return cls._merge_text_segments([*pending_partials, current_text])

    @staticmethod
    def _has_assistant_output(event: VoiceLLMOutputEvent) -> bool:
        return bool(event.transcript or event.audio or event.is_final)

    @staticmethod
    def _clip_text(value: Any, limit: int = 180) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    @classmethod
    def _tool_calls_for_log(cls, calls: list[ToolCall]) -> list[dict[str, Any]]:
        logged: list[dict[str, Any]] = []
        for call in calls:
            args = call.arguments or {}
            logged.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": cls._clip_text(json.dumps(args, ensure_ascii=False, sort_keys=True)),
                }
            )
        return logged

    @classmethod
    def _model_event_kind(cls, event: VoiceLLMOutputEvent) -> str:
        if event.tool_calls:
            return "tool_call"
        if event.user_transcript:
            return "user_transcript"
        if event.barge_in:
            return "turn_started"
        if event.is_final:
            return "assistant_final"
        if event.transcript:
            return "assistant_delta"
        if event.audio is not None:
            return "audio_delta"
        return "event"

    @classmethod
    def _log_model_event(cls, session_id: str, event: VoiceLLMOutputEvent) -> None:
        kind = cls._model_event_kind(event)
        audio = event.audio
        fields: dict[str, Any] = {
            "question_id": event.question_id,
            "reply_id": event.reply_id,
            "is_final": event.is_final,
            "barge_in": event.barge_in,
        }
        if event.user_transcript:
            fields["user_transcript"] = cls._clip_text(event.user_transcript)
        if event.transcript:
            fields["transcript"] = cls._clip_text(event.transcript)
        if audio is not None:
            fields["audio"] = {
                "bytes": len(audio.data or b""),
                "sample_rate": audio.sample_rate,
                "is_final": audio.is_final,
            }
        if event.tool_calls:
            fields["tool_calls"] = cls._tool_calls_for_log(event.tool_calls)
        info_kinds = {"turn_started", "user_transcript", "tool_call", "assistant_final"}
        log = logger.info if kind in info_kinds else logger.debug
        log(
            "persona model event session=%s kind=%s fields=%s",
            session_id or "",
            kind,
            json.dumps(fields, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _accepted_async_task_result() -> dict[str, Any]:
        return {
            "ok": True,
            "accepted": True,
            "status": "accepted",
            "reply": "好的，请稍等，我现在开始处理。",
        }

    @staticmethod
    def _is_terminal_task(task: dict[str, Any]) -> bool:
        return str(task.get("status") or "") in {"completed", "failed", "cancelled"}

    @staticmethod
    def _latest_event_message(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            message = str(event.get("message") or "").strip()
            if message:
                return message
        return ""

    @staticmethod
    def _latest_artifact_id(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            payload = event.get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            if isinstance(payload, dict):
                artifact_id = str(payload.get("artifact_id") or "").strip()
                if artifact_id:
                    return artifact_id
        return ""

    async def _wait_for_task_terminal(self, task_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if self.task_client is None:
            raise RuntimeError("persona task client is not initialized")
        deadline = asyncio.get_running_loop().time() + self.task_monitor_timeout_seconds
        after_seq = 0
        events: list[dict[str, Any]] = []
        task = await self.task_client.get_task(task_id)
        while True:
            new_events = await self.task_client.get_task_events(task_id, after_seq=after_seq, limit=100)
            for event in new_events:
                events.append(event)
                try:
                    after_seq = max(after_seq, int(event.get("seq") or 0))
                except (TypeError, ValueError):
                    pass
            task = await self.task_client.get_task(task_id)
            if self._is_terminal_task(task):
                return task, events
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"task {task_id} did not finish before persona monitor timeout")
            await asyncio.sleep(self.task_poll_interval_seconds)

    def _task_completion_prompt(
        self,
        user_request: str,
        task: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> str:
        status = str(task.get("status") or "").strip()
        summary = str(task.get("result_summary") or "").strip() or self._latest_event_message(events)
        artifact_id = self._latest_artifact_id(events)
        artifact_hint = "资料已经在聊天侧生成，用户可以打开链接查看。" if artifact_id else "没有生成可打开的资料链接。"
        return "\n".join(
            [
                "后台任务结果已经返回。请作为数字人用自然口语回复用户，保持一到两句话。",
                f"用户原始请求：{user_request}",
                f"任务状态：{status}",
                f"结果摘要：{summary or '无'}",
                artifact_hint,
                "不要朗读内部字段名、JSON、任务 ID 或 artifact ID。",
            ]
        )

    @staticmethod
    def _task_start_failed_prompt(user_request: str, error: Exception) -> str:
        return "\n".join(
            [
                "后台任务没有成功启动。请作为数字人用一句自然口语告诉用户稍后再试。",
                f"用户原始请求：{user_request}",
                f"错误原因：{error}",
            ]
        )

    async def _run_async_task(
        self,
        pending: PendingAsyncTask,
        injected: asyncio.Queue[VoiceLLMInputEvent],
    ) -> None:
        if self.task_client is None:
            raise RuntimeError("persona task client is not initialized")
        try:
            task = await self.task_client.create_task(pending.session_id, pending.args)
            task_id = str(task.get("id") or "").strip()
            if not task_id:
                raise RuntimeError("task service did not return a task id")
            final_task, events = await self._wait_for_task_terminal(task_id)
            prompt = self._task_completion_prompt(pending.user_request, final_task, events)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("persona async task failed")
            prompt = self._task_start_failed_prompt(pending.user_request, exc)
        await injected.put(VoiceLLMInputEvent(text=prompt))

    @staticmethod
    def _persona_system_prompt(session_config: VoiceLLMSessionConfig) -> str:
        prompt = (session_config.system_prompt or "").strip()
        if not prompt:
            return PERSONA_AGENT_INSTRUCTIONS
        return f"{PERSONA_AGENT_INSTRUCTIONS}\n\n角色设定：\n{prompt}"

    async def _merged_input_stream(
        self,
        input_stream: AsyncIterator[VoiceLLMInputEvent],
        injected: asyncio.Queue[VoiceLLMInputEvent],
    ) -> AsyncIterator[VoiceLLMInputEvent]:
        source = input_stream.__aiter__()
        source_done = False
        while True:
            try:
                while True:
                    yield injected.get_nowait()
            except asyncio.QueueEmpty:
                pass

            if source_done:
                try:
                    yield await asyncio.wait_for(injected.get(), timeout=0.2)
                    continue
                except asyncio.TimeoutError:
                    return

            try:
                yield await source.__anext__()
            except StopAsyncIteration:
                source_done = True

    async def converse_stream(
        self,
        input_stream: AsyncIterator[VoiceLLMInputEvent],
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> AsyncIterator[VoiceLLMOutputEvent]:
        if self.model_plugin is None:
            raise RuntimeError("persona model plugin is not initialized")
        session_config = session_config or VoiceLLMSessionConfig()
        model_session_config = replace(
            session_config,
            system_prompt=self._persona_system_prompt(session_config),
            tools=PERSONA_TOOL_DEFINITIONS,
        )
        injected: asyncio.Queue[VoiceLLMInputEvent] = asyncio.Queue()
        pending_partials: list[str] = []
        turn_transcripts: list[str] = []
        pending_task_starts: list[PendingAsyncTask] = []
        background_tasks: set[asyncio.Task[None]] = set()

        def schedule_task_start(pending: PendingAsyncTask) -> None:
            task = asyncio.create_task(self._run_async_task(pending, injected))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        try:
            async for event in self.model_plugin.converse_stream(
                self._merged_input_stream(input_stream, injected),
                session_config=model_session_config,
            ):
                self._log_model_event(session_config.session_id, event)
                if event.user_transcript:
                    turn_transcripts.append(event.user_transcript)
                    event = replace(event, user_transcript="")
                    if not event.tool_calls and not event.barge_in and not self._has_assistant_output(event):
                        continue

                if event.tool_calls:
                    for call in event.tool_calls:
                        name = call.name.strip()
                        if name == "wait_for_more_input":
                            partial_text = self._partial_text_for_wait(call, turn_transcripts)
                            if partial_text:
                                pending_partials.append(partial_text)
                            turn_transcripts.clear()
                            try:
                                result = await self._execute_tool(call, session_config.session_id)
                            except Exception as exc:
                                logger.exception("persona wait tool call failed: %s", call.name)
                                result = {"ok": False, "error": str(exc)}
                            await injected.put(
                                VoiceLLMInputEvent(
                                    tool_result=ToolResult(
                                        id=call.id,
                                        name=call.name,
                                        result=result,
                                        suppress_response=True,
                                    )
                                )
                            )
                            continue

                        final_user_text = self._final_user_text(call, pending_partials, turn_transcripts)
                        effective_call = call
                        if name == "create_task" and final_user_text:
                            args = dict(call.arguments or {})
                            args["user_request"] = final_user_text
                            args["kind"] = str(args.get("kind") or "research").strip() or "research"
                            effective_call = ToolCall(id=call.id, name=call.name, arguments=args)
                        if final_user_text:
                            yield VoiceLLMOutputEvent(
                                user_transcript=final_user_text,
                                question_id=event.question_id,
                                reply_id=event.reply_id,
                            )
                        pending_partials.clear()
                        turn_transcripts.clear()

                        if name == "create_task":
                            pending_task_starts.append(
                                PendingAsyncTask(
                                    session_id=session_config.session_id,
                                    args=dict(effective_call.arguments or {}),
                                    user_request=final_user_text,
                                )
                            )
                            result = self._accepted_async_task_result()
                        else:
                            try:
                                result = await self._execute_tool(effective_call, session_config.session_id)
                            except Exception as exc:
                                logger.exception("persona tool call failed: %s", call.name)
                                result = {"ok": False, "error": str(exc)}
                        await injected.put(
                            VoiceLLMInputEvent(
                                tool_result=ToolResult(
                                    id=call.id,
                                    name=call.name,
                                    result=result,
                                )
                            )
                        )
                    continue

                if self._has_assistant_output(event) and (pending_partials or turn_transcripts):
                    final_user_text = self._merge_text_segments([*pending_partials, *turn_transcripts])
                    if final_user_text:
                        yield VoiceLLMOutputEvent(
                            user_transcript=final_user_text,
                            question_id=event.question_id,
                            reply_id=event.reply_id,
                        )
                    pending_partials.clear()
                    turn_transcripts.clear()
                yield event

                if event.is_final and pending_task_starts:
                    starts = pending_task_starts[:]
                    pending_task_starts.clear()
                    for pending in starts:
                        schedule_task_start(pending)
        finally:
            for task in background_tasks:
                task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
