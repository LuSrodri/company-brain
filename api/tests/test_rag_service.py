"""Testes do ``RAGService`` contra um ChromaDB real (sem modelos pesados).

Usa ``MockEmbedding`` para gerar vetores determinísticos e ``llm=None`` (não há
chat aqui). Validam o gerenciamento de documentos: listagem, exclusão completa,
ausência de chunks órfãos no update e o skip por hash de conteúdo.
"""

from __future__ import annotations

import pytest
from llama_index.core import Document
from llama_index.core.embeddings import MockEmbedding

from app.config import get_settings
from app.core.rag import RAGService


@pytest.fixture
def service(tmp_path, monkeypatch) -> RAGService:
    monkeypatch.setenv("CB_CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CB_CHROMA_COLLECTION", "test_collection")
    get_settings.cache_clear()
    settings = get_settings()
    svc = RAGService(
        settings,
        llm=None,
        embed_model=MockEmbedding(embed_dim=8),
        image_describer=None,  # type: ignore[arg-type]
        stt_engine=None,  # type: ignore[arg-type]
    )
    svc.load()
    yield svc
    get_settings.cache_clear()


def _page(doc_id: str, page: int, *, content_hash: str = "H") -> Document:
    """Simula uma página de PDF: doc_id derivado ``doc_id:p{n}`` + metadado base."""
    return Document(
        text=f"conteudo da pagina {page}",
        doc_id=f"{doc_id}:p{page}",
        metadata={
            "document": doc_id,
            "source": f"{doc_id}.pdf",
            "modality": "pdf",
            "page": page,
            "content_hash": content_hash,
        },
    )


def test_reupsert_with_fewer_pages_leaves_no_orphans(service: RAGService) -> None:
    service.upsert_documents([_page("rep", i) for i in (1, 2, 3, 4, 5)], doc_id="rep")
    assert service.count() == 5

    # Nova versão com menos páginas: as antigas :p4/:p5 devem sumir.
    service.upsert_documents([_page("rep", i) for i in (1, 2, 3)], doc_id="rep")
    assert service.count() == 3

    records = service.list_documents()
    assert len(records) == 1
    assert records[0].pages == [1, 2, 3]


def test_delete_removes_all_pages(service: RAGService) -> None:
    service.upsert_documents([_page("rep", i) for i in (1, 2)], doc_id="rep")
    assert service.delete("rep") is True
    assert service.count() == 0
    # Idempotente: apagar de novo não acha nada.
    assert service.delete("rep") is False


def test_list_documents_groups_children_under_base_id(service: RAGService) -> None:
    service.upsert_text("documento textual", doc_id="t1")
    service.upsert_documents([_page("rep", i) for i in (1, 2)], doc_id="rep")

    records = {record.doc_id: record for record in service.list_documents()}
    assert set(records) == {"t1", "rep"}
    assert records["rep"].chunks == 2
    assert records["rep"].pages == [1, 2]
    assert records["rep"].source == "rep.pdf"
    assert records["t1"].modality == "text"


def test_upsert_text_skips_when_unchanged(service: RAGService) -> None:
    first = service.upsert_text("mesmo texto", doc_id="t1")
    assert first.skipped is False

    second = service.upsert_text("mesmo texto", doc_id="t1")
    assert second.skipped is True
    assert service.count() == 1

    third = service.upsert_text("texto diferente", doc_id="t1")
    assert third.skipped is False
    assert service.count() == 1
