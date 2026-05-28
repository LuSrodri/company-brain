"""Endpoint de chat sobre a base de conhecimento."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.core.rag import RAGService
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatResponse:
    """Responde à mensagem usando RAG (recuperação no Chroma + Gemma 4)."""
    result = service.chat(
        payload.message,
        history=[turn.model_dump() for turn in payload.history],
        top_k=payload.top_k,
    )
    return ChatResponse(**result)
