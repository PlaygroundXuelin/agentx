"""High-level helpers for Docling + LlamaIndex PDF ingestion."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, FormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader

LOGGER = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PdfChunk:
    """Normalized representation of a Docling-produced PDF chunk."""

    chunk_id: str
    text: str
    summary: str
    embedding: list[float]
    metadata: dict[str, Any]
    images: tuple[str, ...]


class DoclingPdfPipeline:
    """Pipeline that parses, chunks, summarizes, and embeds PDF content."""

    def __init__(
        self,
        *,
        summary_llm: Any,
        sentence_transformer: str,
        include_images: bool = True,
        artifacts_dir: Path | None = None,
        pdf_options: PdfPipelineOptions | None = None,
        node_parser: DoclingNodeParser | None = None,
    ) -> None:
        self._summary_llm = summary_llm
        self._embed_model = HuggingFaceEmbedding(model_name=sentence_transformer)
        self._include_images = include_images
        self._artifacts_dir = artifacts_dir
        self._base_pdf_options = pdf_options or PdfPipelineOptions()
        self._node_parser = node_parser or DoclingNodeParser()
        self._summary_extractor = SummaryExtractor(llm=summary_llm, summaries=["self"])

    def process(self, pdf_path: str | Path) -> list[PdfChunk]:
        """Parse ``pdf_path`` and return chunked summaries with embeddings."""

        source_path = Path(pdf_path)
        try:
            return self._process_with_options(source_path, include_images=self._include_images)
        except FileNotFoundError as exc:
            if self._include_images:
                LOGGER.warning(
                    "Docling assets missing for %s (%s). Retrying without OCR/image export.",
                    source_path,
                    exc,
                )
                return self._process_with_options(source_path, include_images=False)
            raise

    def _process_with_options(self, source_path: Path, *, include_images: bool) -> list[PdfChunk]:
        documents = list(self._load_docling_documents(source_path, include_images=include_images))

        if not documents:
            return []

        pipeline = IngestionPipeline(
            transformations=[
                self._node_parser,
                self._summary_extractor,
                self._embed_model,
            ]
        )
        nodes = pipeline.run(documents=documents)

        return [self._build_chunk(node) for node in nodes if isinstance(node, TextNode)]

    def _load_docling_documents(self, pdf_path: Path, *, include_images: bool) -> Iterable[Any]:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    pipeline_options=self._configured_pdf_options(pdf_path, include_images),
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_cls=StandardPdfPipeline,
                )
            }
        )
        reader = DoclingReader(
            export_type=DoclingReader.ExportType.JSON,
            doc_converter=converter,
        )
        return reader.load_data(file_path=pdf_path)

    def _configured_pdf_options(self, pdf_path: Path, include_images: bool) -> PdfPipelineOptions:
        if not include_images:
            return self._base_pdf_options.model_copy(
                update={
                    "generate_page_images": False,
                    "generate_picture_images": False,
                    "artifacts_path": None,
                    "do_ocr": False,
                }
            )

        artifacts_root = self._artifacts_dir or pdf_path.parent / f"{pdf_path.stem}_artifacts"
        artifacts_root.mkdir(parents=True, exist_ok=True)
        return self._base_pdf_options.model_copy(
            update={
                "generate_page_images": True,
                "generate_picture_images": True,
                "artifacts_path": str(artifacts_root),
                "do_ocr": True,
            }
        )

    @staticmethod
    def _build_chunk(node: TextNode) -> PdfChunk:
        metadata: dict[str, Any] = dict(node.metadata or {})
        summary = metadata.get("section_summary") or ""
        images = DoclingPdfPipeline._extract_image_paths(metadata)
        embedding_vector = list(node.embedding or [])

        return PdfChunk(
            chunk_id=node.node_id,
            text=node.get_content(metadata_mode=MetadataMode.NONE),
            summary=summary,
            embedding=embedding_vector,
            metadata=metadata,
            images=images,
        )

    @staticmethod
    def _extract_image_paths(metadata: Mapping[str, Any]) -> tuple[str, ...]:
        def iter_image_entries(value: Any) -> Iterator[Mapping[str, Any]]:
            if isinstance(value, Mapping):
                yield value
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    if isinstance(item, Mapping):
                        yield item

        image_entries = metadata.get("images") or metadata.get("figures") or ()
        image_paths: list[str] = []

        for entry in iter_image_entries(image_entries):
            for key in ("path", "image_path", "file_path", "uri"):
                if candidate := entry.get(key):
                    image_paths.append(str(candidate))
                    break

        return tuple(image_paths)
