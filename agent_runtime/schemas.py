from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["queued", "running", "waiting_user", "completed", "failed", "cancelled"]


class Task(BaseModel):
    id: str
    session_id: str
    character_id: str | None = None
    kind: str = "research"
    title: str
    user_request: str
    status: TaskStatus = "queued"
    progress: int = 0
    result_summary: str | None = None
    locale: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TaskEvent(BaseModel):
    event_type: str
    status: TaskStatus = "running"
    message: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    payload: dict[str, Any] | None = None


class ArtifactRequest(BaseModel):
    type: str = "markdown"
    title: str
    mime_type: str = "text/markdown; charset=utf-8"
    content: str
    metadata: dict[str, Any] | None = None
