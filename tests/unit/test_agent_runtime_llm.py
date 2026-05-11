from types import SimpleNamespace

import pytest

from agent_runtime.i18n import Localizer
from agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentLLM, agent_llm_config_from_cyberverse_config
from agent_runtime.schemas import Task


def test_agent_llm_uses_inference_llm_default_config():
    config = {
        "inference": {
            "llm": {
                "default": "qwen",
                "qwen": {
                    "api_key": "dashscope-key",
                    "model": "qwen3.6-plus",
                    "temperature": 0.7,
                    "extra_body": {"enable_thinking": False},
                },
            }
        }
    }

    llm_config = agent_llm_config_from_cyberverse_config(config)

    assert llm_config.provider == "qwen"
    assert llm_config.api_key == "dashscope-key"
    assert llm_config.model == "qwen3.6-plus"
    assert llm_config.temperature == 0.7
    assert llm_config.extra_body == {"enable_thinking": False}


def test_agent_llm_worker_override_keeps_provider_config_defaults():
    config = {
        "inference": {
            "agent_worker": {
                "llm": {
                    "provider": "qwen",
                    "model": "qwen-max",
                    "temperature": 0.1,
                }
            },
            "llm": {
                "default": "qwen",
                "qwen": {
                    "api_key": "dashscope-key",
                    "model": "qwen3.6-plus",
                    "temperature": 0.7,
                    "extra_body": {"enable_thinking": False},
                },
            },
        }
    }

    llm_config = agent_llm_config_from_cyberverse_config(config)

    assert llm_config.provider == "qwen"
    assert llm_config.api_key == "dashscope-key"
    assert llm_config.model == "qwen-max"
    assert llm_config.temperature == 0.1
    assert llm_config.extra_body == {"enable_thinking": False}


class FakeChatCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


@pytest.mark.asyncio
async def test_openai_compatible_agent_llm_runs_graph_node_calls():
    completions = FakeChatCompletions(
        [
            '{"kind":"research","normalized_request":"查今天知乎热点","title":"知乎热点"}',
            '{"steps":["确认查询范围","检索相关信息","整理输出"]}',
            '{"title":"知乎热点","content_markdown":"# 知乎热点\\n\\n内容","summary":"已经整理好知乎热点。"}',
        ]
    )
    llm = OpenAICompatibleAgentLLM(
        AgentLLMConfig(
            provider="qwen",
            model="qwen3.6-plus",
            api_key="fake-key",
            base_url="https://example.com/v1",
            extra_body={"enable_thinking": False},
        )
    )
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    task = Task(
        id="task-1",
        session_id="session-1",
        title="知乎热点",
        user_request="今天知乎有哪些热门信息",
    )
    localizer = Localizer("zh-CN")

    classified = await llm.classify_task(task, localizer)
    steps = await llm.plan_task(task, localizer)
    draft = await llm.draft_artifact(task, [], localizer)

    assert classified["normalized_request"] == "查今天知乎热点"
    assert steps == ["确认查询范围", "检索相关信息", "整理输出"]
    assert draft.summary == "已经整理好知乎热点。"
    assert all(call["model"] == "qwen3.6-plus" for call in completions.calls)
    assert all(call["extra_body"] == {"enable_thinking": False} for call in completions.calls)
