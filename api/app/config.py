"""Configurações da aplicação, carregadas de variáveis de ambiente / `.env`.

Todas as variáveis usam o prefixo ``CB_`` (Company Brain).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Servidor
    host: str = "0.0.0.0"
    port: int = 8000

    # Google AI Studio (Gemini API) — serve o LLM/ingestão multimodal (Gemini).
    # Gere a chave em https://aistudio.google.com/apikey
    google_api_key: str | None = None

    # OpenAI — serve os embeddings (text-embedding-3-large) via API.
    # Gere a chave em https://platform.openai.com/api-keys
    openai_api_key: str | None = None

    # Hugging Face — usado apenas pelo modelo que ainda roda localmente
    # (Whisper STT); o token é opcional para ele.
    hf_token: str | None = None
    hf_cache_dir: str = ".hf_cache"

    # Modelos
    # LLM/ingestão multimodal servido pela API do Google AI Studio (texto+imagem).
    llm_model: str = "gemini-3.1-flash-lite"
    # Embeddings via API da OpenAI (text-embedding-3-large = 3072 dims).
    embed_model: str = "text-embedding-3-large"
    # Dimensão dos embeddings: None = nativa do modelo (3072 p/ 3-large). Defina
    # um valor menor (ex.: 1024) para encurtar via parâmetro `dimensions` da API
    # (Matryoshka) — troca um pouco de acurácia por vetores menores/baratos.
    embed_dimensions: int | None = None
    # Tamanho do lote enviado por requisição à API de embeddings.
    embed_batch_size: int = 100
    # STT (transcrição de áudio): Whisper multilíngue (99 idiomas, inclui PT-BR)
    # servido pelo ecossistema Hugging Face (roda localmente, na GPU/CPU).
    stt_model: str = "openai/whisper-large-v3-turbo"
    # Idioma forçado para a transcrição ("portuguese", "english", ...) ou None
    # para detecção automática do Whisper.
    stt_language: str | None = None
    # Chunking da transcrição. 0 = long-form NATIVO do Whisper (recomendado:
    # timestamps precisos por segmento). >0 ativa a janela deslizante do
    # transformers (mais rápida em áudios muito longos, porém com timestamps
    # grosseiros — colapsa segmentos, cf. aviso "experimental" do transformers).
    stt_chunk_length_s: int = 0
    # Device dos modelos locais (Whisper/embeddings): "auto", "cuda" (NVIDIA ou
    # AMD ROCm), "rocm" (alias de cuda), "mps" (Apple) ou "cpu".
    device: str = "auto"
    # Limite de tokens gerados pelo LLM (max_output_tokens na API do Google).
    # Generoso de propósito: o gemini-3.1-flash-lite sempre raciocina ("thinking") e
    # esse raciocínio consome tokens antes da resposta; um teto baixo termina em
    # finish_reason=MAX_TOKENS (que o wrapper GoogleGenAI trata como erro).
    max_new_tokens: int = 2048

    # Ingestão de documentos
    # DPI usado pelo pdf2image ao rasterizar páginas de PDF para o Gemma (OCR).
    pdf_dpi: int = 200
    # Caminho para os binários do poppler (pdftoppm/pdfinfo). Deixe vazio se o
    # poppler já está no PATH do sistema.
    poppler_path: str | None = None

    # ChromaDB
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "company_brain"

    # RAG
    chunk_size: int = 1024
    chunk_overlap: int = 128
    # top_k da recuperação. Base multi-documento pede recall maior; 8 cobre bem
    # sem estourar contexto/custo (cada chunk ~chunk_size tokens).
    similarity_top_k: int = 8

    # Uploads
    upload_dir: str = "./data/uploads"

    def ensure_dirs(self) -> None:
        """Cria os diretórios locais necessários (chroma, uploads, cache HF)."""
        for path in (self.chroma_path, self.upload_dir, self.hf_cache_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Retorna a instância única (cacheada) de configurações."""
    return Settings()
