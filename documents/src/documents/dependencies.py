"""Dependency providers for the documents service."""

from functools import lru_cache

from documents.services.indexing_service import DocumentIndexService
from documents.services.settings import DocumentSettings

_DOCUMENT_SETTINGS: DocumentSettings | None = None


def configure_document_dependencies(settings: DocumentSettings) -> None:
    """Store application settings so DI providers can build services."""

    global _DOCUMENT_SETTINGS
    _DOCUMENT_SETTINGS = settings
    get_document_index_service.cache_clear()


@lru_cache(maxsize=1)
def get_document_index_service() -> DocumentIndexService:
    """Return a singleton instance of the document index service."""

    if _DOCUMENT_SETTINGS is None:
        raise RuntimeError("Document settings have not been configured.")
    return DocumentIndexService(_DOCUMENT_SETTINGS)
