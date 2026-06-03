"""Embeddings via OpenAI ``text-embedding-3-large`` (API).

Diferente do antigo harrier-oss (local, com instrução só na query), os embeddings
da OpenAI rodam na nuvem e não exigem device/instrução: o mesmo modelo embeda
documentos e queries. A dimensão padrão do ``text-embedding-3-large`` é 3072;
pode ser encurtada via ``embed_dimensions`` (parâmetro ``dimensions`` da API).
"""

from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding

from app.config import Settings


def build_embed_model(settings: Settings) -> BaseEmbedding:
    """Instancia o modelo de embeddings da OpenAI.

    Import de ``OpenAIEmbedding`` é feito aqui dentro para manter o import do
    módulo barato (evita carregar o SDK em testes que mockam a camada de modelos).
    A chave vem de ``CB_OPENAI_API_KEY`` (ou, na falta, do ``OPENAI_API_KEY`` lido
    pelo próprio SDK).
    """
    from llama_index.embeddings.openai import OpenAIEmbedding

    kwargs: dict = {
        "model": settings.embed_model,
        "api_key": settings.openai_api_key,
        "embed_batch_size": settings.embed_batch_size,
    }
    # Só passa `dimensions` quando o usuário pediu encurtamento; None usa a
    # dimensão nativa do modelo (3072 para o text-embedding-3-large).
    if settings.embed_dimensions is not None:
        kwargs["dimensions"] = settings.embed_dimensions

    return OpenAIEmbedding(**kwargs)
