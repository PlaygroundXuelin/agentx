"""Pytest fixtures for the documents API service."""

from collections.abc import Iterable, Iterator

import pytest
from fastapi.testclient import TestClient

from documents.app import create_app
from documents.dependencies import get_document_index_service
from documents.schemas import DocumentPayload, SearchResult
from documents.services.indexing_service import DocumentIndexNotReadyError


class FakeDocumentIndexService:
    """Minimal stand-in for the real index service used in tests."""

    def __init__(self) -> None:
        self.indexed_documents = []
        self.search_calls: list[tuple[str, int]] = []
        self.results: list[SearchResult] = []
        self.raise_not_ready = False

    def index_documents(self, documents: Iterable[DocumentPayload]) -> int:
        self.indexed_documents = list(documents)
        return len(self.indexed_documents)

    def search(self, query: str, *, limit: int) -> list[SearchResult]:
        self.search_calls.append((query, limit))
        if self.raise_not_ready:
            raise DocumentIndexNotReadyError("Document index has not been built yet.")
        return self.results


@pytest.fixture()
def fake_service() -> FakeDocumentIndexService:
    return FakeDocumentIndexService()


@pytest.fixture()
def client(fake_service: FakeDocumentIndexService) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_document_index_service] = lambda: fake_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
