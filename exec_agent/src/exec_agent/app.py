"""FastAPI service for new Agentic RAG subprojects."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Final

import pydantic.dataclasses as pydantic_dataclasses
import structlog
import uvicorn
from core.cmd_utils import load_app_settings
from core.logging import configure_logging
from core.settings import CoreSettings
from fastapi import APIRouter, FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from litellm import aresponses
from pydantic import BaseModel, Field

LOGGER: Final = structlog.get_logger(__name__)


@pydantic_dataclasses.dataclass(frozen=True)
class LlmSettings:
    model: str = ""
    timeout_seconds: int = 30
    temperature: float = 0


@pydantic_dataclasses.dataclass(frozen=True)
class AppSettings(CoreSettings):
    api_prefix: str = "/v1"
    cors_origins: list[str] = dataclasses.field(default_factory=lambda: ["*"])
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    service_name: str = "exec_agent"
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)
    llm: LlmSettings = LlmSettings()


class QueryRequest(BaseModel):
    user_input: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    response: str


def create_app(settings: AppSettings) -> FastAPI:
    """Instantiate a FastAPI application configured with defaults."""

    app = FastAPI(
        title=settings.title or "App Template",
        description=settings.description,
        version=settings.version,
        default_response_class=ORJSONResponse,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app, settings)
    return app


def register_routes(app: FastAPI, settings: AppSettings) -> None:
    """Attach routes that demonstrate the service layout."""

    @app.get("/", include_in_schema=False)
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    router = APIRouter(prefix=settings.api_prefix, tags=["exec_agent"])

    @router.get("/ping")
    def ping() -> dict[str, str]:
        return {
            "message": "pong",
            "service": settings.service_name or "exec_agent",
        }

    def extract_response_text(response: object) -> str:
        if isinstance(response, dict):
            output = response.get("output")
        else:
            output = getattr(response, "output", None)
        if not output:
            return ""

        parts: list[str] = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content")
            else:
                content = getattr(item, "content", None)
            if not content:
                continue
            for chunk in content:
                if isinstance(chunk, dict):
                    chunk_type = chunk.get("type")
                else:
                    chunk_type = getattr(chunk, "type", None)
                if chunk_type == "output_text":
                    text = chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "text", "")
                elif isinstance(chunk, dict):
                    text = chunk.get("text", "")
                else:
                    text = getattr(chunk, "text", "")
                if text:
                    parts.append(str(text))

        return "".join(parts).strip()

    @router.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest) -> QueryResponse:
        try:
            completion = await aresponses(
                input=payload.user_input,
                model=settings.llm.model,
                temperature=settings.llm.temperature,
                timeout=settings.llm.timeout_seconds,
            )
        except Exception as exc:
            LOGGER.exception("exec_agent.query_failed", error=str(exc))
            raise HTTPException(status_code=502, detail="Model request failed") from exc

        response_text = extract_response_text(completion)

        if not response_text:
            raise HTTPException(status_code=502, detail="Model response empty")

        return QueryResponse(response=response_text)

    app.include_router(router)


def serve() -> None:
    """Entrypoint that mirrors other subprojects."""

    settings: AppSettings = load_app_settings(AppSettings, None)

    configure_logging(settings.logging)
    application = create_app(settings)

    LOGGER.info(
        "exec_agent.startup",
        host=settings.host,
        port=settings.port,
        service_name=settings.service_name,
        metadata=settings.metadata,
    )

    config = uvicorn.Config(
        app=application,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
    server = uvicorn.Server(config=config)
    asyncio.run(server.serve())


if __name__ == "__main__":  # pragma: no cover - import-time guard
    serve()
