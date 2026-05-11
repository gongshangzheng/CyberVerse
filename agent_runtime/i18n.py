from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


DEFAULT_LOCALE = "zh-CN"


CATALOGS: dict[str, dict[str, Any]] = {
    "zh-CN": {
        "list.joiner": "、",
        "plan.steps": ["理解问题", "检索信息", "整理资料"],
        "event.plan_created": "我已经拆好任务步骤：{steps_text}。",
        "event.research_blocked": "搜索工具接口已就绪，但真实搜索适配器尚未接入；本次会生成一份占位资料。",
        "event.completed": "我已经整理好资料。第一版搜索接口还是占位实现，后续接入真实搜索后会返回实际来源。",
        "artifact.title_suffix": "研究资料",
        "artifact.user_request": "用户请求",
        "artifact.label_separator": "：",
        "artifact.current_status": "当前状态",
        "artifact.null_search_line_1": "搜索工具抽象已经接入，但真实搜索适配器尚未启用。",
        "artifact.null_search_line_2": "这份资料用于验证 CyberVerse Task Service、LangGraph Worker、事件流和 artifact 交付链路。",
        "artifact.results_intro": "已检索到以下候选信息：",
        "worker.cancelled": "Worker 已停止执行任务。",
        "worker.failed": "Worker 执行失败：{error}",
    },
    "en": {
        "list.joiner": ", ",
        "plan.steps": ["Understand the request", "Search for information", "Prepare materials"],
        "event.plan_created": "I have broken the task into steps: {steps_text}.",
        "event.research_blocked": "The search tool interface is ready, but no real search adapter is enabled yet; I will generate a placeholder artifact for this run.",
        "event.completed": "I have prepared the materials. Search is still using a placeholder in this first version; real sources will appear after a search adapter is connected.",
        "artifact.title_suffix": "Research Notes",
        "artifact.user_request": "User request",
        "artifact.label_separator": ": ",
        "artifact.current_status": "Current status",
        "artifact.null_search_line_1": "The search tool abstraction is connected, but no real search adapter is enabled yet.",
        "artifact.null_search_line_2": "This artifact verifies the CyberVerse Task Service, LangGraph Worker, event stream, and artifact delivery path.",
        "artifact.results_intro": "The following candidate information was found:",
        "worker.cancelled": "The worker has stopped the task.",
        "worker.failed": "Worker execution failed: {error}",
    },
    "ja": {
        "list.joiner": "、",
        "plan.steps": ["依頼を理解する", "情報を検索する", "資料を整理する"],
        "event.plan_created": "タスクを次の手順に分解しました：{steps_text}。",
        "event.research_blocked": "検索ツールのインターフェースは準備済みですが、実検索アダプターはまだ有効化されていません。今回はプレースホルダー資料を生成します。",
        "event.completed": "資料を整理しました。初版では検索はプレースホルダー実装のため、実検索アダプター接続後に実際の出典が返ります。",
        "artifact.title_suffix": "調査資料",
        "artifact.user_request": "ユーザー依頼",
        "artifact.label_separator": "：",
        "artifact.current_status": "現在の状態",
        "artifact.null_search_line_1": "検索ツール抽象は接続されていますが、実検索アダプターはまだ有効化されていません。",
        "artifact.null_search_line_2": "この資料は CyberVerse Task Service、LangGraph Worker、イベントストリーム、artifact 配信経路を検証するためのものです。",
        "artifact.results_intro": "以下の候補情報が見つかりました：",
        "worker.cancelled": "Worker がタスクの実行を停止しました。",
        "worker.failed": "Worker の実行に失敗しました：{error}",
    },
    "ko": {
        "list.joiner": ", ",
        "plan.steps": ["요청 이해", "정보 검색", "자료 정리"],
        "event.plan_created": "작업을 다음 단계로 나누었습니다: {steps_text}.",
        "event.research_blocked": "검색 도구 인터페이스는 준비되었지만 실제 검색 어댑터는 아직 연결되지 않았습니다. 이번에는 자리표시자 자료를 생성합니다.",
        "event.completed": "자료를 정리했습니다. 첫 버전에서는 검색이 자리표시자 구현이며, 실제 검색 어댑터를 연결한 뒤 실제 출처가 반환됩니다.",
        "artifact.title_suffix": "조사 자료",
        "artifact.user_request": "사용자 요청",
        "artifact.label_separator": ": ",
        "artifact.current_status": "현재 상태",
        "artifact.null_search_line_1": "검색 도구 추상화는 연결되었지만 실제 검색 어댑터는 아직 활성화되지 않았습니다.",
        "artifact.null_search_line_2": "이 자료는 CyberVerse Task Service, LangGraph Worker, 이벤트 스트림, artifact 전달 경로를 검증하기 위한 것입니다.",
        "artifact.results_intro": "다음 후보 정보를 찾았습니다:",
        "worker.cancelled": "Worker가 작업 실행을 중지했습니다.",
        "worker.failed": "Worker 실행 실패: {error}",
    },
}


def normalize_locale(locale: str | None) -> str:
    raw = (locale or "").strip()
    if not raw:
        raw = os.getenv("CYBERVERSE_AGENT_LOCALE", DEFAULT_LOCALE)
    lowered = raw.replace("_", "-").lower()
    if lowered in {"zh", "zh-cn", "zh-hans", "cn"}:
        return "zh-CN"
    if lowered.startswith("en"):
        return "en"
    if lowered.startswith("ja") or lowered.startswith("jp"):
        return "ja"
    if lowered.startswith("ko") or lowered.startswith("kr"):
        return "ko"
    return DEFAULT_LOCALE


class Localizer:
    def __init__(self, locale: str | None = None) -> None:
        self.locale = normalize_locale(locale)

    def value(self, key: str) -> Any:
        catalog = CATALOGS.get(self.locale, CATALOGS[DEFAULT_LOCALE])
        if key in catalog:
            return catalog[key]
        return CATALOGS[DEFAULT_LOCALE][key]

    def text(self, key: str, **kwargs: Any) -> str:
        value = self.value(key)
        if not isinstance(value, str):
            raise TypeError(f"i18n key {key!r} is not a string")
        return value.format_map(_SafeFormat(kwargs))

    def list(self, key: str) -> list[str]:
        value = self.value(key)
        if not isinstance(value, list):
            raise TypeError(f"i18n key {key!r} is not a list")
        return [str(item) for item in value]


class _SafeFormat(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def locale_from_metadata(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return normalize_locale(None)
    value = metadata.get("locale") or metadata.get("language")
    return normalize_locale(str(value) if value is not None else None)
