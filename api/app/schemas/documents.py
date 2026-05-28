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
    status: str = "upserted"
    total_chunks: int = Field(..., description="Total de chunks na coleção após a operação.")
