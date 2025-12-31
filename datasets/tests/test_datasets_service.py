from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from core.cmd_utils import load_app_settings
from fastapi.testclient import TestClient

from datasets.app import AppSettings, create_app


def test_app_settings_defaults() -> None:
    settings = AppSettings()

    assert settings.api_prefix == "/v1"
    assert settings.cors_origins == ["*"]
    assert settings.host == "0.0.0.0"
    assert settings.port == 8080
    assert settings.reload is False
    assert settings.service_name == "datasets-service"
    assert settings.metadata == {}
    assert settings.catalog_base_url == ""


def test_load_app_settings_with_env(tmp_path: Path) -> None:
    config_path = (
        Path(__file__).resolve().parent.parent / "src" / "datasets" / "configs" / "local.yaml"
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
    assert settings.metadata == {"owner": "datasets-team"}


@pytest.fixture()
def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "datasets" / "configs" / "local.yaml"


@pytest.fixture()
def client(default_config_path: Path, tmp_path: Path) -> Iterator[TestClient]:
    fs_root = tmp_path / "store"
    original_env = os.environ.copy()
    try:
        os.environ["FS__ROOT"] = str(fs_root)
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
    assert response.json() == {"message": "pong", "service": "datasets-service"}


def test_create_app_populates_catalog_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SVC_CATALOG_URL", "http://127.0.0.1:9191")
    app = create_app(AppSettings())
    assert app.state.catalog_base_url == "http://127.0.0.1:9191"


def test_ping_catalog_success(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    client.app.state.catalog_base_url = "http://catalog:9000"
    captured: dict[str, str] = {}

    class DummyAsyncClient:
        def __init__(self, **_: object) -> None:
            # mirror httpx.AsyncClient signature (timeout, etc.) without storing
            pass

        async def __aenter__(self) -> DummyAsyncClient:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        async def get(self, url: str) -> httpx.Response:
            captured["url"] = url
            return httpx.Response(
                status_code=200,
                json={"message": "pong"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr("datasets.app.httpx.AsyncClient", DummyAsyncClient)

    response = client.get("/v1/ping-catalog")

    assert response.status_code == 200
    assert response.json() == {"message": "pong-catalog"}
    assert captured["url"] == "http://catalog:9000/v1/ping"


def test_ping_catalog_missing_config(client: TestClient) -> None:
    client.app.state.catalog_base_url = ""

    response = client.get("/v1/ping-catalog")

    assert response.status_code == 503
    assert response.json()["detail"] == "Catalog base URL not configured."


def test_ping_catalog_failure(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    client.app.state.catalog_base_url = "http://catalog:9000"

    class FailingAsyncClient:
        def __init__(self, **_: object) -> None:
            pass

        async def __aenter__(self) -> FailingAsyncClient:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        async def get(self, url: str) -> httpx.Response:
            return httpx.Response(
                status_code=500,
                json={"message": "error"},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr("datasets.app.httpx.AsyncClient", FailingAsyncClient)

    response = client.get("/v1/ping-catalog")

    assert response.status_code == 503
    assert response.json()["detail"].startswith("Catalog ping failed with status")


def test_store_metadata_creates_directory(client: TestClient) -> None:
    dataset_id = uuid4()
    metadata = {"name": "example", "tags": ["a", "b"]}

    response = client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": metadata},
    )

    assert response.status_code == 200
    assert response.json() == {"dataset_id": str(dataset_id), "status": "created"}

    store = client.app.state.store
    assert store.dataset_dir_exists(dataset_id) is True
    assert store.fetch_metadata(dataset_id) == metadata


def test_store_metadata_overlay_mode(client: TestClient) -> None:
    dataset_id = uuid4()
    initial_metadata = {"name": "dataset", "tags": ["a"]}

    response = client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": initial_metadata},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "created"

    response = client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": {"tags": ["a", "b"], "size": 10}},
        headers={"X-Metadata-Mode": "overlay"},
    )

    assert response.status_code == 200
    assert response.json() == {"dataset_id": str(dataset_id), "status": "updated"}

    store = client.app.state.store
    assert store.fetch_metadata(dataset_id) == {"name": "dataset", "tags": ["a", "b"], "size": 10}


def test_store_metadata_override_mode(client: TestClient) -> None:
    dataset_id = uuid4()
    client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": {"a": 1, "b": 2}},
    )

    response = client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": {"c": 3}},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "updated"

    store = client.app.state.store
    assert store.fetch_metadata(dataset_id) == {"c": 3}


def test_store_metadata_invalid_mode(client: TestClient) -> None:
    dataset_id = uuid4()
    client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": {"a": 1}},
    )

    response = client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": {"a": 2}},
        headers={"X-Metadata-Mode": "merge"},
    )

    assert response.status_code == 400
    assert "Invalid metadata update mode" in response.json()["detail"]


def test_upload_file_creates_version(client: TestClient) -> None:
    dataset_id = uuid4()
    client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": {"a": 1}},
    )

    response = client.post(
        "/v1/datasets/uploadFile",
        data={"dataset_id": str(dataset_id)},
        files={"file": ("data.bin", b"hello world", "application/octet-stream")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["dataset_id"] == str(dataset_id)
    assert payload["status"] == "uploaded"
    assert payload["filename"].startswith("data-")
    assert payload["filename"].endswith(".bin")

    store = client.app.state.store
    data = store.fetch_data(dataset_id)
    assert data == b"hello world"


def test_upload_file_missing_dataset(client: TestClient) -> None:
    response = client.post(
        "/v1/datasets/uploadFile",
        data={"dataset_id": str(uuid4())},
        files={"file": ("data.bin", b"hello world", "application/octet-stream")},
    )

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_upload_file_multiple_versions(client: TestClient) -> None:
    dataset_id = uuid4()
    client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": {"a": 1}},
    )

    first = client.post(
        "/v1/datasets/uploadFile",
        data={"dataset_id": str(dataset_id)},
        files={"file": ("data.bin", b"first", "application/octet-stream")},
    )
    assert first.status_code == 201

    second = client.post(
        "/v1/datasets/uploadFile",
        data={"dataset_id": str(dataset_id)},
        files={"file": ("data.bin", b"second", "application/octet-stream")},
    )
    assert second.status_code == 201

    store = client.app.state.store
    latest = store.fetch_data(dataset_id)
    assert latest == b"second"

    dataset_root = Path(client.app.state.settings.fs.root)
    dataset_path = dataset_root / str(dataset_id)
    version_files = sorted(dataset_path.glob("data-*.bin"))
    assert len(version_files) == 2
    symlink = dataset_path / "data.bin"
    assert symlink.is_symlink()
    assert symlink.resolve() == version_files[-1]


def test_get_metadata_endpoint(client: TestClient) -> None:
    dataset_id = uuid4()
    metadata = {"name": "example", "tags": ["x"]}
    client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": metadata},
    )

    response = client.get(
        "/v1/datasets/metadata",
        params={"dataset_id": str(dataset_id)},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == str(dataset_id)
    assert body["metadata"] == metadata


def test_get_metadata_missing_dataset(client: TestClient) -> None:
    response = client.get(
        "/v1/datasets/metadata",
        params={"dataset_id": str(uuid4())},
    )

    assert response.status_code == 404


def test_get_file_endpoint(client: TestClient) -> None:
    dataset_id = uuid4()
    client.post(
        "/v1/datasets/storeMetadata",
        json={"dataset_id": str(dataset_id), "metadata": {"a": 1}},
    )
    client.post(
        "/v1/datasets/uploadFile",
        data={"dataset_id": str(dataset_id)},
        files={"file": ("data.bin", b"payload", "application/octet-stream")},
    )

    response = client.get(
        "/v1/datasets/file",
        params={"dataset_id": str(dataset_id)},
    )

    assert response.status_code == 200
    assert response.content == b"payload"
    assert response.headers["content-type"] == "application/octet-stream"


def test_get_file_missing_dataset(client: TestClient) -> None:
    response = client.get(
        "/v1/datasets/file",
        params={"dataset_id": str(uuid4())},
    )

    assert response.status_code == 404
