from agent_runtime.i18n import Localizer, normalize_locale
from agent_runtime.llm import DraftResult
from agent_runtime.graph import _draft_markdown, run_task_with_langgraph
from agent_runtime.schemas import Task
from agent_runtime.tools import MockSearchTool, NullSearchTool


class FakeCallbacks:
    def __init__(self):
        self.events = []
        self.artifacts = []

    async def event(self, task_id, event):
        self.events.append((task_id, event))

    async def artifact(self, task_id, artifact):
        self.artifacts.append((task_id, artifact))
        return {"id": "artifact-1"}


class FakeAgentLLM:
    provider = "fake"
    model = "fake-agent-llm"

    def __init__(self):
        self.calls = []

    async def classify_task(self, task, localizer):
        self.calls.append(("classify_task", task.user_request, localizer.locale))
        return {
            "kind": task.kind or "research",
            "normalized_request": task.user_request,
            "title": task.title,
        }

    async def plan_task(self, task, localizer):
        self.calls.append(("plan_task", task.user_request, localizer.locale))
        return localizer.list("plan.steps")

    async def draft_artifact(self, task, results, localizer):
        self.calls.append(("draft_artifact", task.user_request, len(results), localizer.locale))
        return DraftResult(
            title=f"{task.title} - {localizer.text('artifact.title_suffix')}",
            content_markdown=_draft_markdown(task, results, localizer),
            summary=localizer.text("event.completed"),
        )


def test_normalize_locale_aliases():
    assert normalize_locale("zh") == "zh-CN"
    assert normalize_locale("en-US") == "en"
    assert normalize_locale("ja-JP") == "ja"
    assert normalize_locale("ko-KR") == "ko"


def test_agent_markdown_uses_task_locale():
    task = Task(
        id="task-1",
        session_id="session-1",
        title="Hot topics",
        user_request="What is trending?",
        locale="en",
    )

    content = _draft_markdown(task, [], Localizer(task.locale))

    assert "User request: What is trending?" in content
    assert "Current status" in content
    assert "搜索工具" not in content


async def test_langgraph_research_task_uses_localized_messages(monkeypatch, tmp_path):
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    callbacks = FakeCallbacks()
    task = Task(
        id="task-1",
        session_id="session-1",
        title="Hot topics",
        user_request="What is trending?",
        locale="en",
    )

    llm = FakeAgentLLM()

    await run_task_with_langgraph(task, NullSearchTool(), callbacks, llm=llm)

    assert len(callbacks.events) == 3
    assert [event.event_type for _, event in callbacks.events] == [
        "plan.created",
        "research.blocked",
        "task.completed",
    ]
    assert callbacks.events[-1][1].status == "completed"
    assert "I have prepared the materials" in callbacks.events[-1][1].message
    assert "User request" in callbacks.artifacts[0][1].content
    assert [call[0] for call in llm.calls] == ["classify_task", "plan_task", "draft_artifact"]
    assert callbacks.events[0][1].payload["llm_provider"] == "fake"
    assert callbacks.artifacts[0][1].metadata["llm_model"] == "fake-agent-llm"


async def test_langgraph_research_task_mock_search_success(monkeypatch, tmp_path):
    monkeypatch.setenv("LANGGRAPH_CHECKPOINT_DB", str(tmp_path / "checkpoints.db"))
    callbacks = FakeCallbacks()
    task = Task(
        id="task-1",
        session_id="session-1",
        title="知乎热点",
        user_request="今天知乎有哪些热门信息",
        locale="zh-CN",
    )

    llm = FakeAgentLLM()

    await run_task_with_langgraph(task, MockSearchTool(), callbacks, llm=llm)

    assert [event.event_type for _, event in callbacks.events] == [
        "plan.created",
        "task.completed",
    ]
    assert callbacks.artifacts[0][1].metadata["source_count"] == 1
    assert "Mock search result" in callbacks.artifacts[0][1].content
    assert [call[0] for call in llm.calls] == ["classify_task", "plan_task", "draft_artifact"]
