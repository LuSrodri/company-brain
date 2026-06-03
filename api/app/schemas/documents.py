"""Schemas dos endpoints de documentos."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    """Documento de texto enviado para ingestão/atualização."""

    doc_id: str = Field(..., description="Identificador estável; reenviar atualiza o documento.")
    text: str = Field(..., min_length=1, description="Conteúdo textual do documento.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    """Resposta após upsert de documento(s)."""

    doc_ids: list[str]
    status: str = Field("upserted", description="'upserted' ou 'unchanged' (conteúdo idêntico).")
    total_chunks: int = Field(..., description="Total de chunks na coleção após a operação.")


class DocumentInfo(BaseModel):
    """Visão agregada de um documento lógico indexado."""

    doc_id: str = Field(..., description="Identificador base do documento.")
    source: str | None = Field(None, description="Nome do arquivo/fonte original.")
    modality: str | None = Field(None, description="text, pdf, image, audio, spreadsheet, document.")
    pages: list[int] = Field(default_factory=list, description="Páginas/abas indexadas.")
    chunks: int = Field(..., description="Quantidade de chunks (nós) deste documento.")
    content_hash: str | None = Field(None, description="SHA-256 do conteúdo ingerido.")


class DocumentList(BaseModel):
    """Listagem dos documentos presentes na base de conhecimento."""

    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int
