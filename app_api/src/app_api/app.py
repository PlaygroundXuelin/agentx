from __future__ import annotations

import asyncio
import dataclasses

import pydantic.dataclasses as pydantic_dataclasses
import uvicorn
from core import configure_logging
from core.cmd_utils import load_app_settings
from core.settings import CoreSettings
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from app_api.schemas import QueryRequest, QueryResponse, StreamError
from app_api.services.rag import full_answer, stream_answer


@pydantic_dataclasses.dataclass(frozen=True)
class AppSettings(CoreSettings):
    api_prefix: str = "/v1"
    cors_origins: list[str] = dataclasses.field(default_factory=lambda: ["*"])
    sse_ping_seconds: int = 15
    sse_send_timeout_seconds: int = 30
    host: str = "0.0.0.0"
    port: int = 8000


def create_app(settings: AppSettings) -> FastAPI:
    """Instantiate the FastAPI application with the provided settings."""

    app = FastAPI(
        title=settings.title,
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

    app.state.settings = settings
    register_routes(app, settings)
    return app


def register_routes(app: FastAPI, settings: AppSettings) -> None:
    """Attach HTTP routes that close over the resolved settings."""

    @app.get("/", include_in_schema=False)
    async def root():  # noqa: D401 - simple health check
        return {"status": "ok"}

    @app.post(f"{settings.api_prefix}/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        return await full_answer(req)

    @app.get(f"{settings.api_prefix}/query/stream")
    async def query_stream(
        request: Request,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
        temperature: float = 0.0,
    ):
        req = QueryRequest(
            query=query,
            session_id=session_id,
            top_k=top_k,
            temperature=temperature,
        )

        async def event_gen():
            try:
                i = 0
                async for event in stream_answer(req):
                    if await request.is_disconnected():
                        break
                    payload = event.model_dump()
                    event_type = payload.get("type")

                    if event_type == "chunk":
                        i += 1
                        payload["index"] = i
                        yield JSONServerSentEvent(payload)
                    elif event_type == "final":
                        yield JSONServerSentEvent(payload)
                    else:
                        yield JSONServerSentEvent(
                            {
                                "type": "error",
                                "message": (
                                    f"unknown event type: {event_type or type(event).__name__}"
                                ),
                            }
                        )
            except Exception as exc:  # pylint: disable=broad-except
                error_payload = StreamError(type="error", message=str(exc)).model_dump()
                yield JSONServerSentEvent(error_payload)

        return EventSourceResponse(
            event_gen(),
            ping=settings.sse_ping_seconds,
            send_timeout=settings.sse_send_timeout_seconds,
            headers={
                "Cache-Control": "no-cache",
                # Helpful for Nginx/Traefik to avoid buffering SSE
                "X-Accel-Buffering": "no",
            },
        )


def serve() -> None:
    """Expose an app runner compatible with setuptools entry points."""

    settings: AppSettings = load_app_settings(AppSettings, None)

    configure_logging(settings.logging)

    application = create_app(settings)

    config = uvicorn.Config(
        app=application,
        host=settings.host,
        port=settings.port,
        reload=False,
    )
    server = uvicorn.Server(config=config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    serve()
