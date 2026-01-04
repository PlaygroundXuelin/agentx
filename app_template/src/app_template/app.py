"""FastAPI service."""

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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

LOGGER: Final = structlog.get_logger(__name__)


@pydantic_dataclasses.dataclass(frozen=True)
class AppSettings(CoreSettings):
    api_prefix: str = "/v1"
    cors_origins: list[str] = dataclasses.field(default_factory=lambda: ["*"])
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    service_name: str = "app-template"
    metadata: dict[str, str] = dataclasses.field(default_factory=dict)


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

    router = APIRouter(prefix=settings.api_prefix, tags=["app-template"])

    @router.get("/ping")
    def ping() -> dict[str, str]:
        return {
            "message": "pong",
            "service": settings.service_name or "app-template",
        }

    app.include_router(router)


def serve() -> None:
    """Entrypoint that mirrors other subprojects."""

    settings: AppSettings = load_app_settings(AppSettings, None)

    configure_logging(settings.logging)
    application = create_app(settings)

    LOGGER.info(
        "app_template.startup",
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
