import asyncio
import urllib.error
import urllib.request

import pytest

from agent_runtime.callbacks import CallbackClient


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self.body


@pytest.mark.asyncio
async def test_callback_client_retries_5xx_then_succeeds(monkeypatch):
    calls = {"count": 0}

    def fake_urlopen(req: urllib.request.Request, timeout: int):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.HTTPError(req.full_url, 500, "server error", {}, None)
        return FakeResponse(b'{"ok": true}')

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    client = CallbackClient(server_url="http://server/api/v1", internal_token="token")
    result = await client._post_json("/internal/tasks/t1/events", {"event_type": "x"}, timeout=1)

    assert result == {"ok": True}
    assert calls["count"] == 3


@pytest.mark.asyncio
async def test_callback_client_treats_409_as_terminal(monkeypatch):
    def fake_urlopen(req: urllib.request.Request, timeout: int):
        raise urllib.error.HTTPError(req.full_url, 409, "terminal", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = CallbackClient(server_url="http://server/api/v1")
    result = await client._post_json("/internal/tasks/t1/events", {"event_type": "x"}, timeout=1)

    assert result == {"status": "terminal"}
