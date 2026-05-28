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

    # Hugging Face
    hf_token: str | None = None
    hf_cache_dir: str = ".hf_cache"

    # Modelos
    llm_model: str = "google/gemma-4-E2B-it"
    embed_model: str = "microsoft/harrier-oss-v1-0.6b"
    device: str = "auto"
    dtype: str = "auto"
    enable_thinking: bool = False
    max_new_tokens: int = 1024

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
