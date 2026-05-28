"""Dependências compartilhadas das rotas."""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from app.core.rag import RAGService


def get_rag_service(request: Request) -> RAGService:
    """Recupera o ``RAGService`` inicializado no ``lifespan`` da aplicação.

    Nos testes, esta dependência é sobrescrita via ``app.dependency_overrides``
    por um *fake*, evitando o download dos modelos.
    """
    service: RAGService | None = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service ainda não está pronto.",
        )
    return service
