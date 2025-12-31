"""Utilities for persisting and indexing uploaded PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import Final
from uuid import uuid4

import pydantic.dataclasses as pydantic_dataclasses
import structlog
from fastapi import UploadFile
from pypdf import PdfReader

from documents.schemas import DocumentPayload
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

    extracted_text = extract_text_from_pdf(file_path)
    if not extracted_text:
        LOGGER.warning("No text extracted from %s", file_path)
        return

    metadata_base = {
        "source_path": str(file_path),
    }
    if original_filename:
        metadata_base["original_filename"] = original_filename

    payloads = [
        DocumentPayload(
            document_id=document_id,
            content=extracted_text,
            metadata=metadata_base,
        )
    ]

    try:
        service.index_documents(payloads)
        LOGGER.info(
            "Indexed PDF document %s from %s",
            document_id,
            file_path,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to index document %s: %s", document_id, exc)


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract plain text from a PDF file."""

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.exception("Failed to read PDF %s: %s", file_path, exc)
        return ""

    chunks: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to extract text from page in %s: %s", file_path, exc)
            continue
        if text:
            chunks.append(text)

    return "\n".join(chunks).strip()
