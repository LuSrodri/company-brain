"""Fixtures de teste.

Os modelos (Gemma 4 / harrier) **não** são carregados: a dependência
``get_rag_service`` é sobrescrita por um ``FakeRAGService`` em memória, e o
``TestClient`` é usado SEM o gerenciador de contexto, de modo que o ``lifespan``
(que baixaria os modelos) nunca executa.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient


class FakeRAGService:
    """Implementação em memória do contrato usado pelas rotas."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.chat_calls: list[dict[str, Any]] = []

    # --- ingestão ---
    def upsert_text(self, text: str, *, doc_id: str, metadata: dict | None = None) -> str:
        self.docs[doc_id] = {"text": text, "metadata": metadata or {}}
        return doc_id

    def ingest_file(self, path: str, *, doc_id: str, metadata: dict | None = None) -> list[str]:
        self.docs[doc_id] = {"path": path, "metadata": metadata or {}}
        return [doc_id]

    def delete(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)

    def count(self) -> int:
        return len(self.docs)

    # --- chat ---
    def chat(
        self, message: str, *, history: list | None = None, top_k: int | None = None
    ) -> dict[str, Any]:
        self.chat_calls.append({"message": message, "history": history, "top_k": top_k})
        sources = [
            {"text": d.get("text", ""), "score": 0.9, "metadata": d["metadata"]}
            for d in self.docs.values()
        ]
        return {"answer": f"Resposta simulada para: {message}", "sources": sources}


@pytest.fixture(scope="session", autouse=True)
def _isolate_paths(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Redireciona diretórios locais para uma pasta temporária."""
    base = tmp_path_factory.mktemp("company_brain")
    os.environ["CB_UPLOAD_DIR"] = str(base / "uploads")
    os.environ["CB_CHROMA_PATH"] = str(base / "chroma")
    os.environ["CB_HF_CACHE_DIR"] = str(base / "hf")

    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def fake_service() -> FakeRAGService:
    return FakeRAGService()


@pytest.fixture
def client(fake_service: FakeRAGService) -> TestClient:
    from app.api.deps import get_rag_service
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_rag_service] = lambda: fake_service
    # Sem 'with': o lifespan (carregamento de modelos) não é executado.
    return TestClient(app)
