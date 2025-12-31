"""Utilities for persisting and indexing uploaded PDF documents."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final
from uuid import uuid4

import pydantic.dataclasses as pydantic_dataclasses
import structlog
from fastapi import UploadFile
from llama_index.llms.openai import OpenAI

from documents.schemas import DocumentPayload
from documents.services.docling_pdf_pipeline import DoclingPdfPipeline, PdfChunk
from documents.services.indexing_service import DocumentIndexService
from documents.services.settings import DocumentSettings

LOGGER: Final = structlog.get_logger(__name__)


@pydantic_dataclasses.dataclass(frozen=True)
class DocumentsStore:
    settings: DocumentSettings

    def __post_init__(self):
        destination_dir = self.get_upload_directory()
        destination_dir.mkdir(parents=True, exist_ok=True)

    def get_upload_directory(self) -> Path:
        """Return the directory used to store uploaded files."""

        return Path(self.settings.store.settings.path)

    async def persist_pdf_upload(
        self,
        upload: UploadFile,
        *,
        document_id: str | None = None,
    ) -> tuple[str, Path]:
        """Persist the uploaded PDF to disk and return its document id and path."""

        doc_id = document_id or str(uuid4())
        destination_dir = self.get_upload_directory()

        original_suffix = Path(upload.filename or "").suffix or ".pdf"
        target_path = destination_dir / f"{doc_id}{original_suffix}"

        content = await upload.read()
        if not content:
            raise ValueError("Uploaded PDF is empty.")

        target_path.write_bytes(content)
        await upload.close()

        return doc_id, target_path


def process_pdf_for_indexing(
    file_path: Path,
    *,
    document_id: str,
    service: DocumentIndexService,
    original_filename: str | None,
    document_settings: DocumentSettings,
) -> None:
    """Extract content from the PDF and index it with the provided service."""

    try:
        pipeline = _get_docling_pipeline(document_settings)
        chunks = pipeline.process(file_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to parse PDF %s: %s", file_path, exc)
        return

    if not chunks:
        LOGGER.warning("Docling returned no content for %s", file_path)
        return

    metadata_base = {
        "source_path": str(file_path),
    }
    if original_filename:
        metadata_base["original_filename"] = original_filename

    payloads = [
        _chunk_to_payload(
            document_id=document_id,
            index=index,
            chunk=chunk,
            metadata_base=metadata_base,
        )
        for index, chunk in enumerate(chunks)
    ]

    try:
        service.index_documents(payloads)
        LOGGER.info(
            "Indexed PDF document %s from %s with %d chunks",
            document_id,
            file_path,
            len(payloads),
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to index document %s: %s", document_id, exc)


def _chunk_to_payload(
    *,
    document_id: str,
    index: int,
    chunk: PdfChunk,
    metadata_base: dict[str, str],
) -> DocumentPayload:
    metadata = dict(chunk.metadata or {})
    metadata.update(metadata_base)
    metadata["parent_document_id"] = document_id
    metadata["chunk_index"] = index
    metadata["chunk_summary"] = chunk.summary
    metadata["images"] = list(chunk.images)
    if chunk.embedding:
        metadata["embedding"] = chunk.embedding

    chunk_document_id = f"{document_id}::chunk-{index:04d}"
    return DocumentPayload(
        document_id=chunk_document_id,
        content=chunk.text,
        metadata=metadata,
    )


def _get_docling_pipeline(settings: DocumentSettings) -> DoclingPdfPipeline:
    return _cached_pipeline(
        summary_model=settings.summary_model_name,
        embedding_model=settings.embed.model_name,
    )


def _build_summary_llm(model_name: str):
    if model_name.startswith("openai/"):
        return OpenAI(model=model_name.split("/", 1)[1])
    raise ValueError(f"Unsupported summary model '{model_name}'")


@lru_cache(maxsize=2)
def _cached_pipeline(
    *,
    summary_model: str,
    embedding_model: str,
) -> DoclingPdfPipeline:
    return DoclingPdfPipeline(
        summary_llm=_build_summary_llm(summary_model),
        sentence_transformer=embedding_model,
        include_images=True,
    )
