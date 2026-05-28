"""Serviço de RAG: orquestra LlamaIndex + ChromaDB + Gemma 4.

Responsável por:
    * manter o índice vetorial persistido no ChromaDB local;
    * fazer *upsert* (inserir/atualizar) de documentos por ``doc_id``;
    * responder ao chat usando recuperação + Gemma 4.
"""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import Document
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from app.config import Settings
from app.core.gemma import GemmaEngine
from app.core.ingestion import build_document_from_file, build_text_document

logger = logging.getLogger(__name__)


class RAGService:
    """Orquestra ingestão e chat sobre a base de conhecimento."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: Any,
        embed_model: Any,
        engine: GemmaEngine,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._embed_model = embed_model
        self._engine = engine
        self._index: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------ #
    # Inicialização
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Configura LlamaIndex e abre/cria a coleção persistente do Chroma."""
        import chromadb
        from llama_index.core import Settings as LISettings
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore

        s = self._settings

        LISettings.llm = self._llm
        LISettings.embed_model = self._embed_model
        LISettings.chunk_size = s.chunk_size
        LISettings.chunk_overlap = s.chunk_overlap

        db = chromadb.PersistentClient(path=s.chroma_path)
        self._collection = db.get_or_create_collection(s.chroma_collection)

        vector_store = ChromaVectorStore(chroma_collection=self._collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=self._embed_model,
        )
        logger.info(
            "RAGService pronto (coleção=%s, documentos=%d)",
            s.chroma_collection,
            self.count(),
        )

    @property
    def index(self) -> Any:
        if self._index is None:
            raise RuntimeError("RAGService.load() precisa ser chamado antes do uso.")
        return self._index

    # ------------------------------------------------------------------ #
    # Ingestão (upsert)
    # ------------------------------------------------------------------ #
    def upsert_documents(self, documents: list[Document]) -> list[str]:
        """Insere ou atualiza documentos por ``doc_id`` (idempotente)."""
        ids: list[str] = []
        for doc in documents:
            self._delete_if_exists(doc.doc_id)
            self.index.insert(doc)
            ids.append(doc.doc_id)
        return ids

    def upsert_text(
        self, text: str, *, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        doc = build_text_document(text, doc_id=doc_id, metadata=metadata)
        return self.upsert_documents([doc])[0]

    def ingest_file(
        self, path: str, *, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> str:
        doc = build_document_from_file(
            path, engine=self._engine, doc_id=doc_id, metadata=metadata
        )
        return self.upsert_documents([doc])[0]

    def delete(self, doc_id: str) -> None:
        self._delete_if_exists(doc_id)

    def count(self) -> int:
        """Número de chunks (nós) armazenados na coleção."""
        if self._collection is None:
            return 0
        return self._collection.count()

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #
    def chat(
        self,
        message: str,
        *,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Responde a uma mensagem usando recuperação + Gemma 4.

        Retorna a resposta e as fontes (chunks) recuperadas.
        """
        chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            llm=self._llm,
            similarity_top_k=top_k or self._settings.similarity_top_k,
        )
        chat_history = self._to_chat_history(history or [])
        response = chat_engine.chat(message, chat_history=chat_history)

        sources = [
            {
                "text": node.node.get_content(),
                "score": node.score,
                "metadata": node.node.metadata,
            }
            for node in getattr(response, "source_nodes", [])
        ]
        return {"answer": str(response), "sources": sources}

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _delete_if_exists(self, doc_id: str) -> None:
        try:
            self.index.delete_ref_doc(doc_id, delete_from_docstore=False)
        except Exception:  # noqa: BLE001 — doc inexistente é esperado no insert inicial
            logger.debug("Nenhum documento anterior com doc_id=%s para remover", doc_id)

    @staticmethod
    def _to_chat_history(history: list[dict[str, str]]) -> list[ChatMessage]:
        role_map = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
        }
        return [
            ChatMessage(
                role=role_map.get(item.get("role", "user"), MessageRole.USER),
                content=item.get("content", ""),
            )
            for item in history
        ]
