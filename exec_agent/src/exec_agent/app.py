"""FastAPI service for new Agentic RAG subprojects."""

from __future__ import annotations

import asyncio
from typing import Final

import structlog
import uvicorn
from core.cmd_utils import load_app_settings
from core.logging import configure_logging
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from exec_agent.agent.runner import AgentRunner, LiteLLMChatClient
from exec_agent.infra.config import AppSettings
from exec_agent.tools.executor import ToolExecutor
from exec_agent.tools.impl.retrieve import RetrieveTool
from exec_agent.tools.policies import ToolPolicy
from exec_agent.tools.registry import ToolRegistry

LOGGER: Final = structlog.get_logger(__name__)


class QueryRequest(BaseModel):
    user_input: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    response: str


def build_agent_runner(settings: AppSettings) -> AgentRunner:
    registry = ToolRegistry()
    enabled = set(settings.tools.enabled_tools)
    if "retrieve" in enabled:
        registry.register(RetrieveTool(settings.retrieval))

    policy = ToolPolicy(allowed_tools=enabled, max_calls=settings.tools.max_calls)
    executor = ToolExecutor(registry, policy)
    client = LiteLLMChatClient(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
        timeout_seconds=settings.llm.timeout_seconds,
    )
    return AgentRunner(client=client, registry=registry, executor=executor)


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

    app.state.agent_runner = build_agent_runner(settings)
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

    @router.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest) -> QueryResponse:
        try:
            runner: AgentRunner = app.state.agent_runner
            result = await runner.run(payload.user_input)
        except Exception as exc:
            LOGGER.exception("exec_agent.query_failed", error=str(exc))
            raise HTTPException(status_code=502, detail="Model request failed") from exc

        if not result.output:
            raise HTTPException(status_code=502, detail="Model response empty")

        return QueryResponse(response=result.output)

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
