"""Configuration helpers for the exec agent service."""

from __future__ import annotations

import dataclasses

import pydantic.dataclasses as pydantic_dataclasses
from core.settings import CoreSettings


@pydantic_dataclasses.dataclass(frozen=True)
class LlmSettings:
    model: str = ""
    timeout_seconds: int = 30
    temperature: float = 0


@pydantic_dataclasses.dataclass(frozen=True)
class RetrievalSettings:
    search_paths: list[str] = dataclasses.field(default_factory=lambda: ["documents", "README.md"])
    file_globs: list[str] = dataclasses.field(default_factory=lambda: ["**/*.md", "**/*.txt"])
    max_results: int = 5
    max_snippet_chars: int = 200
    max_file_size_kb: int = 256


@pydantic_dataclasses.dataclass(frozen=True)
class ToolSettings:
    enabled_tools: list[str] = dataclasses.field(default_factory=lambda: ["retrieve"])
    max_calls: int = 8


@pydantic_dataclasses.dataclass(frozen=True)
class AppSettings(CoreSettings):
    api_prefix: str = "/v1"
    cors_origins: list[str] = dataclasses.field(default_factory=lambda: ["*"])
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    service_name: str = "app-template"
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)
    llm: LlmSettings = LlmSettings()
    tools: ToolSettings = ToolSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
