"""Document ingestion endpoints."""

from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from documents.dependencies import get_document_index_service
from documents.schemas import DocumentUploadResponse, IndexDocumentsRequest, IndexDocumentsResponse
from documents.services.indexing_service import DocumentIndexService
from documents.services.pdf_ingestion import DocumentsStore, process_pdf_for_indexing
from documents.services.settings import DocumentSettings


def create_indexing_router(document_settings: DocumentSettings) -> APIRouter:
    router = APIRouter(prefix="/documents", tags=["documents"])

    documents_store = DocumentsStore(settings=document_settings)

    ServiceDependency = Annotated[DocumentIndexService, Depends(get_document_index_service)]
    UploadFileDependency = Annotated[UploadFile, File(...)]
    DocumentIdForm = Annotated[str | None, Form()]

    @router.post("/index", response_model=IndexDocumentsResponse, summary="Index documents")
    async def index_documents(
        request: IndexDocumentsRequest,
        service: ServiceDependency,
    ) -> IndexDocumentsResponse:
        """Ingest new documents and rebuild the LlamaIndex vector store."""

        indexed_count = service.index_documents(request.documents)
        return IndexDocumentsResponse(indexed_count=indexed_count)

    @router.post(
        "/index/pdf",
        response_model=DocumentUploadResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Upload a PDF for asynchronous indexing",
    )
    async def index_pdf_document(
        background_tasks: BackgroundTasks,
        service: ServiceDependency,
        file: UploadFileDependency,
        document_id: DocumentIdForm = None,
    ) -> DocumentUploadResponse:
        """Persist a PDF upload then extract and index its contents asynchronously."""

        if file.content_type not in {"application/pdf", "application/x-pdf"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF uploads are supported.",
            )

        try:
            resolved_id, file_path = await documents_store.persist_pdf_upload(
                file, document_id=document_id
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        background_tasks.add_task(
            process_pdf_for_indexing,
            file_path,
            document_id=resolved_id,
            service=service,
            original_filename=file.filename,
            document_settings=document_settings,
        )

        return DocumentUploadResponse(
            document_id=resolved_id,
            file_path=str(file_path),
            status="accepted",
        )

    return router
