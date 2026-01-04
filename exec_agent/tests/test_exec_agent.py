from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from core.cmd_utils import load_app_settings
from fastapi.testclient import TestClient

from exec_agent.app import AppSettings, create_app


def test_app_settings_defaults() -> None:
    settings = AppSettings()

    assert settings.api_prefix == "/v1"
    assert settings.cors_origins == ["*"]
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.reload is False
    assert settings.service_name == "app-template"
    assert settings.metadata == {}


def test_load_app_settings_with_env(tmp_path: Path) -> None:
    config_path = (
        Path(__file__).resolve().parent.parent / "src" / "exec_agent" / "configs" / "local.yaml"
    )

    env_path = tmp_path / "local.env"
    env_path.write_text("SERVICE_NAME=from-env\n", encoding="utf-8")

    original_env = os.environ.copy()
    try:
        settings = load_app_settings(
            AppSettings,
            ["--config", str(config_path), "--env", str(env_path)],
        )
    finally:
        os.environ.clear()
        os.environ.update(original_env)

    assert settings.service_name == "from-env"
    assert settings.metadata == {"owner": "exec_agent"}


@pytest.fixture()
def default_config_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent / "src" / "exec_agent" / "configs" / "local.yaml"
    )


@pytest.fixture()
def client(default_config_path: Path) -> Iterator[TestClient]:
    original_env = os.environ.copy()
    try:
        settings = load_app_settings(
            AppSettings,
            ["--config", str(default_config_path)],
        )
    finally:
        os.environ.clear()
        os.environ.update(original_env)

    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health_route(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ping_endpoint(client: TestClient) -> None:
    response = client.get("/v1/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "pong", "service": "app-template"}
