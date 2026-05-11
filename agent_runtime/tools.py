from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchTool(Protocol):
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        ...


class NullSearchTool:
    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return []


class MockSearchTool:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self.results = results or [
            SearchResult(
                title="Mock result",
                url="https://example.com/mock",
                snippet="Mock search result for PersonaAgent task tests.",
            )
        ]

    async def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return self.results[:limit]
