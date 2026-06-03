"""Endpoints de ingestão/atualização de documentos.

* ``POST /documents``        -> upsert de documento textual (JSON).
* ``POST /documents/upload`` -> upsert multimodal (texto, pdf, imagem, áudio, xlsx, docx).
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.deps import get_rag_service
from app.config import get_settings
from app.core.ingestion import EmptyDocumentError, SUPPORTED_EXTS, UnsupportedFileTypeError
from app.core.rag import RAGService
from app.schemas.documents import (
    DocumentIn,
    DocumentInfo,
    DocumentList,
    DocumentResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def _status(skipped: bool) -> str:
    return "unchanged" if skipped else "upserted"


@router.get("", response_model=DocumentList)
def list_documents(
    service: Annotated[RAGService, Depends(get_rag_service)],
) -> DocumentList:
    """Lista os documentos indexados na base de conhecimento."""
    records = service.list_documents()
    documents = [
        DocumentInfo(
            doc_id=record.doc_id,
            source=record.source,
            modality=record.modality,
            pages=record.pages,
            chunks=record.chunks,
            content_hash=record.content_hash,
        )
        for record in records
    ]
    return DocumentList(
        documents=documents,
        total_documents=len(documents),
        total_chunks=service.count(),
    )


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upsert_document(
    payload: DocumentIn,
    service: Annotated[RAGService, Depends(get_rag_service)],
) -> DocumentResponse:
    """Insere ou atualiza um documento textual identificado por ``doc_id``."""
    result = service.upsert_text(payload.text, doc_id=payload.doc_id, metadata=payload.metadata)
    return DocumentResponse(
        doc_ids=result.doc_ids,
        status=_status(result.skipped),
        total_chunks=service.count(),
    )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    service: Annotated[RAGService, Depends(get_rag_service)],
    file: Annotated[
        UploadFile,
        File(description="Arquivo: texto, pdf, imagem, áudio, xlsx ou docx."),
    ],
    doc_id: Annotated[str | None, Form()] = None,
    metadata: Annotated[str | None, Form(description="JSON opcional de metadados.")] = None,
) -> DocumentResponse:
    """Ingestão multimodal: texto direto; PDF/imagem/xlsx/docx via Gemma 4; áudio via Whisper."""
    settings = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Extensão não suportada: {ext!r}. Suportadas: {sorted(SUPPORTED_EXTS)}",
        )

    meta: dict[str, Any] = _parse_metadata(metadata)
    resolved_id = doc_id or file.filename or uuid.uuid4().hex

    dest = Path(settings.upload_dir) / f"{uuid.uuid4().hex}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = service.ingest_file(str(dest), doc_id=resolved_id, metadata=meta)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    finally:
        dest.unlink(missing_ok=True)

    return DocumentResponse(
        doc_ids=result.doc_ids,
        status=_status(result.skipped),
        total_chunks=service.count(),
    )


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: str,
    service: Annotated[RAGService, Depends(get_rag_service)],
) -> None:
    """Remove um documento inteiro (todas as páginas/abas) pelo ``doc_id`` base."""
    if not service.delete(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Documento não encontrado: {doc_id!r}")


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            422,
            detail=f"metadata não é um JSON válido: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(422, detail="metadata deve ser um objeto JSON.")
    return parsed
