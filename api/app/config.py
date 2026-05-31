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

    # Google AI Studio (Gemini API) — serve o LLM/ingestão multimodal (Gemma 4).
    # Gere a chave em https://aistudio.google.com/apikey
    google_api_key: str | None = None

    # Hugging Face — usado apenas pelos modelos que ainda rodam localmente
    # (Whisper STT e embeddings harrier-oss); o token é opcional para esses.
    hf_token: str | None = None
    hf_cache_dir: str = ".hf_cache"

    # Modelos
    # LLM/ingestão multimodal servido pela API do Google AI Studio (texto+imagem).
    llm_model: str = "gemma-4-31b-it"
    embed_model: str = "microsoft/harrier-oss-v1-0.6b"
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
    # Generoso de propósito: o gemma-4-31b-it sempre raciocina ("thinking") e
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
    similarity_top_k: int = 4

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
