from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_runtime.i18n import Localizer
from agent_runtime.schemas import Task


_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")


@dataclass
class DraftResult:
    title: str
    content_markdown: str
    summary: str


class AgentLLM(Protocol):
    provider: str
    model: str

    async def classify_task(self, task: Task, localizer: Localizer) -> dict[str, Any]:
        ...

    async def plan_task(self, task: Task, localizer: Localizer) -> list[str]:
        ...

    async def draft_artifact(
        self,
        task: Task,
        results: list[dict[str, str]],
        localizer: Localizer,
    ) -> DraftResult:
        ...


@dataclass
class AgentLLMConfig:
    provider: str = "qwen"
    model: str = "qwen3.6-plus"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)


def _clean_config_string(value: Any) -> str:
    text = str(value or "").strip()
    if _ENV_PLACEHOLDER_RE.match(text):
        return ""
    return text


def _optional_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dashscope_base_url() -> str:
    try:
        from inference.plugins.qwen_endpoint import dashscope_base_url

        return dashscope_base_url()
    except Exception:
        return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def agent_llm_config_from_env() -> AgentLLMConfig:
    provider = _clean_config_string(os.getenv("AGENT_LLM_PROVIDER")) or "qwen"
    model = _clean_config_string(os.getenv("AGENT_LLM_MODEL")) or (
        "qwen3.6-plus" if provider == "qwen" else "gpt-4o"
    )
    api_key = _clean_config_string(os.getenv("AGENT_LLM_API_KEY"))
    if not api_key:
        api_key = _clean_config_string(
            os.getenv("DASHSCOPE_API_KEY") if provider == "qwen" else os.getenv("OPENAI_API_KEY")
        )
    base_url = _clean_config_string(os.getenv("AGENT_LLM_BASE_URL"))
    if not base_url and provider == "qwen":
        base_url = _dashscope_base_url()
    return AgentLLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=_optional_float(os.getenv("AGENT_LLM_TEMPERATURE"), 0.2),
        extra_body={"enable_thinking": False} if provider == "qwen" else {},
    )


def agent_llm_config_from_cyberverse_config(config: dict[str, Any] | None) -> AgentLLMConfig:
    if not isinstance(config, dict):
        return agent_llm_config_from_env()

    inference = config.get("inference", {})
    if not isinstance(inference, dict):
        return agent_llm_config_from_env()

    worker_conf = inference.get("agent_worker", {})
    worker_llm = worker_conf.get("llm", {}) if isinstance(worker_conf, dict) else {}
    worker_llm = worker_llm if isinstance(worker_llm, dict) else {}

    llm_section = inference.get("llm", {})
    llm_section = llm_section if isinstance(llm_section, dict) else {}
    provider = _clean_config_string(worker_llm.get("provider")) or _clean_config_string(llm_section.get("default")) or "qwen"
    provider_conf = llm_section.get(provider, {})
    provider_conf = provider_conf if isinstance(provider_conf, dict) else {}
    merged = {**provider_conf, **worker_llm}

    model = _clean_config_string(merged.get("model")) or ("qwen3.6-plus" if provider == "qwen" else "gpt-4o")
    api_key = _clean_config_string(merged.get("api_key"))
    if not api_key:
        api_key = _clean_config_string(
            os.getenv("DASHSCOPE_API_KEY") if provider == "qwen" else os.getenv("OPENAI_API_KEY")
        )
    base_url = _clean_config_string(merged.get("base_url"))
    if not base_url and provider == "qwen":
        base_url = _dashscope_base_url()
    extra_body = merged.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {"enable_thinking": False} if provider == "qwen" else {}

    return AgentLLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=_optional_float(merged.get("temperature"), 0.2),
        extra_body=extra_body,
    )


class OpenAICompatibleAgentLLM:
    def __init__(self, config: AgentLLMConfig) -> None:
        self.provider = config.provider
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.temperature = config.temperature
        self.extra_body = config.extra_body
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(f"agent LLM api_key is not configured for provider {self.provider!r}")
        from openai import AsyncOpenAI

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def _complete(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = await self._get_client().chat.completions.create(**kwargs)
        if not response.choices:
            return ""
        return str(response.choices[0].message.content or "").strip()

    @staticmethod
    def _json_from_text(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
        if not candidate.startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start >= 0 and end > start:
                candidate = candidate[start : end + 1]
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("agent LLM response is not a JSON object")
        return parsed

    async def _complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return self._json_from_text(await self._complete(messages, temperature=0.1))

    async def classify_task(self, task: Task, localizer: Localizer) -> dict[str, Any]:
        return await self._complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the CyberVerse LangGraph task classifier. "
                        "Return JSON only with keys: kind, normalized_request, title. "
                        "Only kind='research' is supported in this MVP."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "locale": localizer.locale,
                            "kind_hint": task.kind or "research",
                            "title_hint": task.title,
                            "user_request": task.user_request,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )

    async def plan_task(self, task: Task, localizer: Localizer) -> list[str]:
        payload = await self._complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the planning node in a CyberVerse LangGraph research agent. "
                        "Return JSON only: {\"steps\":[...]} with 3 to 5 short, concrete steps. "
                        "Use the requested locale."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "locale": localizer.locale,
                            "task_title": task.title,
                            "user_request": task.user_request,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        steps = payload.get("steps")
        if not isinstance(steps, list):
            return []
        return [str(step).strip() for step in steps if str(step).strip()]

    async def draft_artifact(
        self,
        task: Task,
        results: list[dict[str, str]],
        localizer: Localizer,
    ) -> DraftResult:
        payload = await self._complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the drafting node in a CyberVerse LangGraph research agent. "
                        "Return JSON only with keys: title, content_markdown, summary. "
                        "content_markdown must be a complete Markdown artifact. "
                        "summary must be one short sentence for the user. "
                        "If search results are empty, clearly say the realtime search adapter is not configured yet."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "locale": localizer.locale,
                            "task_title": task.title,
                            "user_request": task.user_request,
                            "search_results": results,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        title = str(payload.get("title") or task.title).strip()
        content = str(payload.get("content_markdown") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        if not content:
            raise ValueError("agent LLM returned empty content_markdown")
        return DraftResult(title=title, content_markdown=content + "\n", summary=summary or localizer.text("event.completed"))


def build_agent_llm(config: AgentLLMConfig | None = None) -> OpenAICompatibleAgentLLM:
    return OpenAICompatibleAgentLLM(config or agent_llm_config_from_env())


def build_agent_llm_from_runtime_config(config: dict[str, Any] | None = None) -> OpenAICompatibleAgentLLM:
    return OpenAICompatibleAgentLLM(agent_llm_config_from_cyberverse_config(config))
