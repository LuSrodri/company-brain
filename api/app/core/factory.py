"""Fábrica que monta o :class:`RAGService` completo a partir das configurações.

Mantida separada de ``deps.py`` para que os testes possam construir um serviço
real ou trocá-lo por um *fake* sem importar a stack pesada de modelos.
"""

from __future__ import annotations

from app.config import Settings
from app.core.embeddings import build_embed_model
from app.core.rag import RAGService
from app.core.transcription import WhisperEngine
from app.core.vision import ImageDescriber

# Janela de contexto do gemini-3.1-flash-lite (série Gemini 3 = ~1M tokens).
LLM_CONTEXT_WINDOW = 1_000_000
# Amostragem genérica (aceita por Gemini 3 e Gemma); ajustável por modelo se preciso.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64


def build_rag_service(settings: Settings) -> RAGService:
    """Monta o LLM GoogleGenAI (gemini-3.1-flash-lite), Whisper, embeddings e o RAGService.

    O mesmo LLM serve o chat e a descrição de imagens da ingestão (via
    :class:`ImageDescriber`) — por isso o modelo precisa ser multimodal
    (texto+imagem). O Whisper só carrega pesos no primeiro uso.
    """
    from google.genai import types
    from llama_index.llms.google_genai import GoogleGenAI

    # Passar context_window E max_tokens evita que o GoogleGenAI faça uma chamada
    # de rede no init para buscar a metadata do modelo.
    llm = GoogleGenAI(
        model=settings.llm_model,
        api_key=settings.google_api_key,
        context_window=LLM_CONTEXT_WINDOW,
        max_tokens=settings.max_new_tokens,
        generation_config=types.GenerateContentConfig(
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            top_k=DEFAULT_TOP_K,
            max_output_tokens=settings.max_new_tokens,
            # Raciocínio mínimo: prioriza latência (o Flash-Lite é otimizado para
            # velocidade) mantendo o "thinking" que os modelos Gemini 3 exigem.
            # Se um teto baixo de tokens for todo consumido pelo raciocínio, o
            # GoogleGenAI levanta erro (finish_reason=MAX_TOKENS) — por isso
            # CB_MAX_NEW_TOKENS é generoso.
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        ),
    )
    image_describer = ImageDescriber(llm)
    stt_engine = WhisperEngine(settings)
    embed_model = build_embed_model(settings)
    return RAGService(
        settings,
        llm=llm,
        embed_model=embed_model,
        image_describer=image_describer,
        stt_engine=stt_engine,
    )
