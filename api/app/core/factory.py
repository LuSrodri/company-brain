"""Fábrica que monta o :class:`RAGService` completo a partir das configurações.

Mantida separada de ``deps.py`` para que os testes possam construir um serviço
real ou trocá-lo por um *fake* sem importar a stack pesada de modelos.
"""

from __future__ import annotations

from app.config import Settings
from app.core.embeddings import build_embed_model
from app.core.gemma import GemmaEngine
from app.core.llm import GemmaLLM
from app.core.rag import RAGService
from app.core.transcription import WhisperEngine


def build_rag_service(settings: Settings) -> RAGService:
    """Instancia engines Gemma/Whisper, LLM, embeddings e o RAGService (sem carregar pesos)."""
    engine = GemmaEngine(settings)
    stt_engine = WhisperEngine(settings)
    llm = GemmaLLM(engine, settings)
    embed_model = build_embed_model(settings)
    return RAGService(
        settings,
        llm=llm,
        embed_model=embed_model,
        engine=engine,
        stt_engine=stt_engine,
    )
