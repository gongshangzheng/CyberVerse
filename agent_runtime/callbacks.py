from __future__ import annotations

import os
import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from agent_runtime.schemas import ArtifactRequest, TaskEvent


class CallbackClient:
    def __init__(self, server_url: str | None = None, internal_token: str | None = None) -> None:
        self.server_url = (server_url or os.getenv("CYBERVERSE_SERVER_URL") or "http://localhost:8080/api/v1").rstrip("/")
        self.internal_token = internal_token if internal_token is not None else os.getenv("AGENT_INTERNAL_TOKEN", "")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.internal_token:
            headers["Authorization"] = f"Bearer {self.internal_token}"
        return headers

    async def _post_json(self, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
        url = f"{self.server_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(4):
            try:
                def do_request() -> dict[str, Any]:
                    req = urllib.request.Request(url, data=body, method="POST", headers=self._headers())
                    with urllib.request.urlopen(req, timeout=timeout) as response:
                        raw = response.read()
                    if not raw:
                        return {}
                    return json.loads(raw.decode("utf-8"))

                return await asyncio.to_thread(do_request)
            except urllib.error.HTTPError as exc:
                if exc.code == 409:
                    return {"status": "terminal"}
                if exc.code != 429 and exc.code < 500:
                    raise
                last_error = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc

            if attempt < 3:
                await asyncio.sleep(0.25 * (2 ** attempt))

        assert last_error is not None
        raise last_error

    async def event(self, task_id: str, event: TaskEvent) -> None:
        await self._post_json(
            f"/internal/tasks/{task_id}/events",
            event.model_dump(exclude_none=True),
            timeout=20,
        )

    async def artifact(self, task_id: str, artifact: ArtifactRequest) -> dict[str, Any]:
        return await self._post_json(
            f"/internal/tasks/{task_id}/artifacts",
            artifact.model_dump(exclude_none=True),
            timeout=30,
        )
