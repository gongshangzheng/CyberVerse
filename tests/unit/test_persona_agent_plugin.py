import logging

import pytest

from inference.core.types import (
    AudioChunk,
    PluginConfig,
    ToolCall,
    VoiceLLMInputEvent,
    VoiceLLMOutputEvent,
    VoiceLLMSessionConfig,
)
from inference.plugins.voice_llm.base import VoiceLLMPlugin
from inference.plugins.voice_llm.persona_agent import PERSONA_AGENT_INSTRUCTIONS, PersonaAgentPlugin


class FakeOmniPlugin(VoiceLLMPlugin):
    name = "omni.fake"

    async def initialize(self, config):
        self.scenario = config.params.get("scenario", "chat")

    async def shutdown(self):
        pass

    async def check_voice(self, session_config=None):
        pass

    async def _next_tool_result(self, input_stream):
        async for event in input_stream:
            if event.tool_result:
                return event.tool_result
        raise AssertionError("expected tool result")

    async def _next_text(self, input_stream):
        async for event in input_stream:
            if event.text:
                return event.text
        raise AssertionError("expected injected text")

    async def _next_input(self, input_stream):
        async for event in input_stream:
            return event
        raise AssertionError("expected input event")

    async def converse_stream(self, input_stream, session_config=None):
        self.last_session_config = session_config
        async for _ in input_stream:
            break

        if self.scenario == "chat":
            yield VoiceLLMOutputEvent(user_transcript="你好")
            yield VoiceLLMOutputEvent(
                transcript="你好，我在。",
                audio=AudioChunk(data=b"audio", sample_rate=16000, is_final=True),
                is_final=True,
            )
            return

        if self.scenario == "wait_for_more_input":
            yield VoiceLLMOutputEvent(user_transcript="我觉得...")
            yield VoiceLLMOutputEvent(
                tool_calls=[
                    ToolCall(
                        id="wait-1",
                        name="wait_for_more_input",
                        arguments={"partial_text": "我觉得...", "reason": "unfinished thought"},
                    )
                ]
            )
            async for event in input_stream:
                if event.tool_result:
                    assert event.tool_result.name == "wait_for_more_input"
                    assert event.tool_result.suppress_response is True
                    return

        if self.scenario == "partial_create_task":
            yield VoiceLLMOutputEvent(user_transcript="我想让你帮我查一下")
            yield VoiceLLMOutputEvent(
                tool_calls=[
                    ToolCall(
                        id="wait-1",
                        name="wait_for_more_input",
                        arguments={"partial_text": "我想让你帮我查一下", "reason": "missing object"},
                    )
                ]
            )
            wait_result = await self._next_tool_result(input_stream)
            assert wait_result.name == "wait_for_more_input"
            assert wait_result.suppress_response is True
            await self._next_input(input_stream)
            yield VoiceLLMOutputEvent(user_transcript="今天知乎有哪些热门信息")
            yield VoiceLLMOutputEvent(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="create_task",
                        arguments={"user_request": "今天知乎有哪些热门信息", "title": "知乎热点"},
                    )
                ]
            )
            tool_result = await self._next_tool_result(input_stream)
            assert tool_result.name == "create_task"
            assert tool_result.result["accepted"] is True
            yield VoiceLLMOutputEvent(
                transcript="好的，请稍等。",
                audio=AudioChunk(data=b"ack", sample_rate=16000, is_final=True),
                is_final=True,
            )
            final_prompt = await self._next_text(input_stream)
            assert "用户原始请求：我想让你帮我查一下今天知乎有哪些热门信息" in final_prompt
            assert "任务状态：completed" in final_prompt
            yield VoiceLLMOutputEvent(
                transcript="查好了，资料已经整理好。",
                audio=AudioChunk(data=b"done", sample_rate=16000, is_final=True),
                is_final=True,
            )
            return

        tool_name = self.scenario
        transcript_by_tool = {
            "create_task": "今天知乎有哪些热门信息",
            "get_task_status": "现在查得怎么样了",
            "cancel_task": "算了不用查了",
        }
        if tool_name in transcript_by_tool:
            yield VoiceLLMOutputEvent(user_transcript=transcript_by_tool[tool_name])
        yield VoiceLLMOutputEvent(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name=tool_name,
                    arguments={"user_request": "今天知乎有哪些热门信息", "title": "知乎热点"},
                )
            ]
        )
        tool_result = await self._next_tool_result(input_stream)
        assert tool_result.name == tool_name
        if tool_name == "create_task":
            assert tool_result.result["accepted"] is True
            yield VoiceLLMOutputEvent(
                transcript="好的，请稍等。",
                audio=AudioChunk(data=b"ack", sample_rate=16000, is_final=True),
                is_final=True,
            )
            final_prompt = await self._next_text(input_stream)
            assert "用户原始请求：今天知乎有哪些热门信息" in final_prompt
            assert "任务状态：completed" in final_prompt
            yield VoiceLLMOutputEvent(
                transcript="查好了，资料已经整理好。",
                audio=AudioChunk(data=b"done", sample_rate=16000, is_final=True),
                is_final=True,
            )
            return
        yield VoiceLLMOutputEvent(
            transcript=f"{tool_name} ok",
            audio=AudioChunk(data=b"ok", sample_rate=16000, is_final=True),
            is_final=True,
        )
        return


class FakeTaskClient:
    def __init__(self):
        self.calls = []

    async def create_task(self, session_id, args):
        self.calls.append(("create_task", session_id, args))
        return {"id": "task-1", "status": "queued"}

    async def get_task(self, task_id):
        self.calls.append(("get_task", task_id, {}))
        return {
            "id": task_id,
            "status": "completed",
            "progress": 100,
            "result_summary": "我已经整理好资料。",
        }

    async def get_task_events(self, task_id, after_seq=0, limit=100):
        self.calls.append(("get_task_events", task_id, {"after_seq": after_seq, "limit": limit}))
        if after_seq > 0:
            return []
        return [
            {
                "task_id": task_id,
                "seq": 1,
                "event_type": "task.completed",
                "status": "completed",
                "message": "我已经整理好资料。",
                "progress": 100,
                "payload": {"artifact_id": "artifact-1"},
            }
        ]

    async def get_task_status(self, session_id):
        self.calls.append(("get_task_status", session_id, {}))
        return {"task": {"id": "task-1", "status": "running", "progress": 30}, "events": []}

    async def cancel_task(self, session_id):
        self.calls.append(("cancel_task", session_id, {}))
        return {"cancelled": True, "task": {"id": "task-1", "status": "cancelled"}}


async def make_persona(scenario, checkpoint_db_path):
    plugin = PersonaAgentPlugin()
    await plugin.initialize(
        PluginConfig(
            plugin_name="persona.persona",
            params={
                "model_provider": "fake",
                "checkpoint_db_path": str(checkpoint_db_path),
                "task_poll_interval_seconds": 0.1,
                "task_monitor_timeout_seconds": 2,
            },
            shared={
                "omni": {
                    "fake": {
                        "plugin_class": "tests.unit.test_persona_agent_plugin.FakeOmniPlugin",
                        "scenario": scenario,
                    }
                }
            },
        )
    )
    plugin.task_client = FakeTaskClient()
    return plugin


async def one_input():
    yield VoiceLLMInputEvent(audio=b"pcm")


async def two_inputs():
    yield VoiceLLMInputEvent(audio=b"pcm-1")
    yield VoiceLLMInputEvent(audio=b"pcm-2")


@pytest.mark.asyncio
async def test_persona_agent_passthrough_chat(tmp_path):
    plugin = await make_persona("chat", tmp_path / "persona.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[0].user_transcript == "你好"
    assert outputs[1].transcript == "你好，我在。"
    assert outputs[1].audio is not None
    assert plugin.model_plugin.last_session_config.tools[0].name == "wait_for_more_input"
    assert plugin.model_plugin.last_session_config.tools[1].name == "create_task"
    assert "PersonaAgent" in plugin.model_plugin.last_session_config.system_prompt
    assert "JSON" not in PERSONA_AGENT_INSTRUCTIONS
    assert plugin.task_client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["get_task_status", "cancel_task"])
async def test_persona_agent_executes_hidden_tool_calls(tool_name, tmp_path):
    plugin = await make_persona(tool_name, tmp_path / f"{tool_name}.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[-1].transcript == f"{tool_name} ok"
    assert outputs[0].user_transcript
    assert plugin.task_client.calls[0][0] == tool_name
    assert plugin.task_client.calls[0][1] == "session-1"


@pytest.mark.asyncio
async def test_persona_agent_create_task_acks_then_runs_async_task(tmp_path):
    plugin = await make_persona("create_task", tmp_path / "create_task.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[0].user_transcript == "今天知乎有哪些热门信息"
    assert outputs[1].transcript == "好的，请稍等。"
    assert outputs[-1].transcript == "查好了，资料已经整理好。"
    assert plugin.task_client.calls[0] == (
        "create_task",
        "session-1",
        {
            "user_request": "今天知乎有哪些热门信息",
            "title": "知乎热点",
            "kind": "research",
        },
    )
    assert any(call[0] == "get_task_events" for call in plugin.task_client.calls)


@pytest.mark.asyncio
async def test_persona_agent_wait_tool_suppresses_output(tmp_path):
    plugin = await make_persona("wait_for_more_input", tmp_path / "wait.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs == []
    assert plugin.task_client.calls == []


def test_persona_event_logs_keep_stream_deltas_out_of_info(caplog):
    logger_name = "inference.plugins.voice_llm.persona_agent"

    with caplog.at_level(logging.INFO, logger=logger_name):
        PersonaAgentPlugin._log_model_event(
            "session-1",
            VoiceLLMOutputEvent(transcript="收"),
        )
        PersonaAgentPlugin._log_model_event(
            "session-1",
            VoiceLLMOutputEvent(
                transcript="收到",
                audio=AudioChunk(data=b"", sample_rate=24000, is_final=True),
                is_final=True,
            ),
        )
        PersonaAgentPlugin._log_model_event(
            "session-1",
            VoiceLLMOutputEvent(audio=AudioChunk(data=b"pcm", sample_rate=24000)),
        )

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "kind=assistant_final" in messages[0]
    assert "收到" in messages[0]


@pytest.mark.asyncio
async def test_persona_agent_merges_waited_partial_into_create_task(tmp_path):
    plugin = await make_persona("partial_create_task", tmp_path / "partial.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                two_inputs(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[0].user_transcript == "我想让你帮我查一下今天知乎有哪些热门信息"
    assert outputs[1].transcript == "好的，请稍等。"
    assert outputs[-1].transcript == "查好了，资料已经整理好。"
    assert plugin.task_client.calls[0] == (
        "create_task",
        "session-1",
        {
            "user_request": "我想让你帮我查一下今天知乎有哪些热门信息",
            "title": "知乎热点",
            "kind": "research",
        },
    )
