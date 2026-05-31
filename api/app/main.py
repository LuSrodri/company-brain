"""Ponto de entrada da API do Company Brain."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app import __version__
from app.api.routes import chat, documents, health
from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Inicializa o RAGService (Gemma via Google AI Studio + harrier + Chroma) no startup."""
    settings = get_settings()
    settings.ensure_dirs()

    # Import tardio: evita carregar a stack de modelos quando o app é só importado
    # (ex.: testes que sobrescrevem a dependência).
    from app.core.factory import build_rag_service

    logger.info("Inicializando RAGService...")
    service = build_rag_service(settings)
    service.load()
    app.state.rag_service = service
    logger.info("RAGService pronto.")
    try:
        yield
    finally:
        app.state.rag_service = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="Company Brain API",
        version=__version__,
        description="RAG multimodal com gemma-4-31b-it (Google AI Studio), harrier-oss embeddings e ChromaDB.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    return app


app = create_app()


def main() -> None:
    """Sobe o servidor com Uvicorn (uso: ``python -m app.main``)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
