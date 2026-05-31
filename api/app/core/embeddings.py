"""Embeddings com ``microsoft/harrier-oss-v1-0.6b`` via LlamaIndex + HuggingFace.

O harrier-oss exige, conforme o model card, que **apenas a query** receba uma
instrução no formato ``Instruct: {tarefa}\\nQuery: {texto}``; os documentos são
embeddados sem instrução. O `HuggingFaceEmbedding` do LlamaIndex aplica
`query_instruction` somente no caminho de busca, o que reproduz exatamente esse
comportamento.
"""

from __future__ import annotations

from llama_index.core.base.embeddings.base import BaseEmbedding

from app.config import Settings
from app.core.devices import resolve_device

# Instrução recomendada pelo model card do harrier-oss para recuperação.
RETRIEVAL_TASK = "Given a web search query, retrieve relevant passages that answer the query"
QUERY_INSTRUCTION = f"Instruct: {RETRIEVAL_TASK}\nQuery: "


def build_embed_model(settings: Settings) -> BaseEmbedding:
    """Instancia o modelo de embeddings harrier-oss.

    Import de `HuggingFaceEmbedding` é feito aqui dentro para manter o import do
    módulo barato (evita carregar torch/sentence-transformers em testes que
    fazem mock da camada de modelos).
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Resolve "auto"/"rocm" em um device concreto (cuda/mps/cpu) compartilhado
    # com o Whisper — cobre NVIDIA, AMD ROCm e fallback CPU.
    device = resolve_device(settings.device)

    return HuggingFaceEmbedding(
        model_name=settings.embed_model,
        device=device,
        cache_folder=settings.hf_cache_dir,
        # harrier-oss: instrução só na query, documentos sem instrução.
        query_instruction=QUERY_INSTRUCTION,
        text_instruction=None,
        trust_remote_code=True,
    )
