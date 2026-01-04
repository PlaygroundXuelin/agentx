"""Simple filesystem retrieval tool."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exec_agent.infra.config import RetrievalSettings
from exec_agent.tools.base import ToolResult, ToolSpec


@dataclass(slots=True)
class RetrieveTool:
    """Search configured paths for a query string and return matching snippets."""

    settings: RetrievalSettings
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="retrieve",
            description="Search local text files for a query and return matching snippets.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "Maximum matches to return."},
                },
                "required": ["query"],
            },
        )
    )

    async def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult.failure(self.spec.name, "missing_query")

        max_results = int(arguments.get("max_results") or self.settings.max_results)
        max_results = max(1, min(max_results, self.settings.max_results))

        matches: list[dict[str, Any]] = []
        for path in self._iter_paths():
            for line_no, line in self._iter_matches(path, query):
                matches.append(
                    {
                        "path": str(path),
                        "line": line_no,
                        "text": line,
                    }
                )
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        payload = {"query": query, "matches": matches}
        content = json.dumps(payload, ensure_ascii=True)
        return ToolResult.from_data(self.spec.name, payload, content=content)

    def _iter_paths(self) -> list[Path]:
        roots = [Path(path).expanduser() for path in self.settings.search_paths]
        paths: list[Path] = []
        for root in roots:
            if root.is_file():
                paths.append(root)
                continue
            if not root.exists():
                continue
            for pattern in self.settings.file_globs:
                paths.extend(root.glob(pattern))

        max_bytes = self.settings.max_file_size_kb * 1024
        filtered: list[Path] = []
        for path in paths:
            if not path.is_file():
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            filtered.append(path)
        return filtered

    def _iter_matches(self, path: Path, query: str) -> list[tuple[int, str]]:
        query_lower = query.casefold()
        matches: list[tuple[int, str]] = []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return matches
        for idx, line in enumerate(lines, start=1):
            if query_lower in line.casefold():
                snippet = line.strip()
                if len(snippet) > self.settings.max_snippet_chars:
                    snippet = snippet[: self.settings.max_snippet_chars].rstrip()
                matches.append((idx, snippet))
                if len(matches) >= self.settings.max_results:
                    break
        return matches
