"""Serviço de RAG: orquestra LlamaIndex + ChromaDB + Gemini (Google AI Studio).

Responsável por:
    * manter o índice vetorial persistido no ChromaDB local;
    * fazer *upsert* (inserir/atualizar) de documentos por ``doc_id``;
    * responder ao chat usando recuperação + Gemini.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from llama_index.core import Document
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from app.config import Settings
from app.core.ingestion import build_documents_from_file, build_text_document
from app.core.transcription import WhisperEngine
from app.core.vision import ImageDescriber

logger = logging.getLogger(__name__)

_HASH_CHUNK = 1 << 20  # 1 MiB

# Orienta o LLM a citar a origem dos trechos recuperados. Os metadados de cada
# nó (source, page, sheet, modality) entram no contexto, e áudios trazem o
# timestamp [HH:MM:SS] no próprio texto — então o modelo tem o que citar.
CITATION_SYSTEM_PROMPT = (
    "Você é o assistente da base de conhecimento da empresa. Use apenas as "
    "informações do contexto recuperado e responda no mesmo idioma da pergunta.\n"
    "Sempre que afirmar algo, cite a origem entre parênteses, com os campos "
    "disponíveis:\n"
    "- nome do arquivo (metadado 'source');\n"
    "- página (metadado 'page') ou aba (metadado 'sheet'), quando houver;\n"
    "- para áudio, o timestamp [HH:MM:SS] que aparece no próprio trecho;\n"
    "- quando ajudar, cite o trecho exato entre aspas.\n"
    "Exemplos: (Relatório.pdf, p. 3) · (Notas.xlsx, aba 'Notas') · "
    "(Reunião.mp3, 00:00:51).\n"
    "Se a informação não estiver no contexto, diga claramente que não a "
    "encontrou na base. Nunca invente fontes nem dados."
)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class UpsertResult:
    """Resultado de uma ingestão/atualização.

    ``skipped`` é ``True`` quando o conteúdo é idêntico ao já indexado (mesmo
    hash) e o reprocessamento caro (OCR/STT/embeddings) foi evitado.
    """

    doc_ids: list[str]
    skipped: bool = False


@dataclass
class DocumentRecord:
    """Visão agregada de um documento lógico (agrupado pelo ``doc_id`` base)."""

    doc_id: str
    doc_ids: list[str] = field(default_factory=list)
    source: str | None = None
    modality: str | None = None
    pages: list[int] = field(default_factory=list)
    chunks: int = 0
    content_hash: str | None = None


class RAGService:
    """Orquestra ingestão e chat sobre a base de conhecimento."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: Any,
        embed_model: Any,
        image_describer: ImageDescriber,
        stt_engine: WhisperEngine,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._embed_model = embed_model
        self._image_describer = image_describer
        self._stt_engine = stt_engine
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
    def upsert_documents(self, documents: list[Document], *, doc_id: str) -> list[str]:
        """Substitui TODOS os chunks do documento base ``doc_id`` pelos novos.

        A limpeza prévia é feita por metadado (``document == doc_id``), o que
        remove também páginas/abas derivadas (``doc_id:p2`` etc.). Isso evita
        chunks órfãos quando a nova versão tem menos páginas que a anterior.
        """
        self._delete_by_document(doc_id)
        ids: list[str] = []
        for doc in documents:
            self.index.insert(doc)
            ids.append(doc.doc_id)
        return ids

    def upsert_text(
        self, text: str, *, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> UpsertResult:
        """Insere/atualiza um documento textual; pula se o conteúdo não mudou."""
        content_hash = _hash_text(text)
        record = self._document_record(doc_id)
        if record is not None and record.content_hash == content_hash:
            return UpsertResult(doc_ids=record.doc_ids, skipped=True)

        meta: dict[str, Any] = {
            **(metadata or {}),
            "source": doc_id,
            "document": doc_id,
            "modality": "text",
            "page": 1,
            "content_hash": content_hash,
        }
        doc = build_text_document(text, doc_id=doc_id, metadata=meta)
        ids = self.upsert_documents([doc], doc_id=doc_id)
        return UpsertResult(doc_ids=ids, skipped=False)

    def ingest_file(
        self, path: str, *, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> UpsertResult:
        """Ingestão de um arquivo. PDFs viram 1 documento por página.

        Se o arquivo for idêntico ao já indexado (mesmo hash de bytes), a
        ingestão é pulada — evitando rodar OCR/STT/embeddings de novo.
        """
        content_hash = _hash_file(path)
        record = self._document_record(doc_id)
        if record is not None and record.content_hash == content_hash:
            return UpsertResult(doc_ids=record.doc_ids, skipped=True)

        documents = build_documents_from_file(
            path,
            engine=self._image_describer,
            stt_engine=self._stt_engine,
            doc_id=doc_id,
            metadata={**(metadata or {}), "content_hash": content_hash},
        )
        ids = self.upsert_documents(documents, doc_id=doc_id)
        return UpsertResult(doc_ids=ids, skipped=False)

    def delete(self, doc_id: str) -> bool:
        """Remove todos os chunks do documento base ``doc_id``.

        Retorna ``True`` se algo foi removido, ``False`` se o documento não
        existia (exclusão idempotente).
        """
        return self._delete_by_document(doc_id)

    def count(self) -> int:
        """Número de chunks (nós) armazenados na coleção."""
        if self._collection is None:
            return 0
        return self._collection.count()

    # ------------------------------------------------------------------ #
    # Listagem / registro de documentos
    # ------------------------------------------------------------------ #
    def list_documents(self) -> list[DocumentRecord]:
        """Lista os documentos lógicos indexados, agrupados pelo ``doc_id`` base."""
        if self._collection is None:
            return []
        result = self._collection.get(include=["metadatas"])
        grouped: dict[str, list[dict[str, Any]]] = {}
        for meta in result.get("metadatas") or []:
            key = meta.get("document")
            if key is None:
                continue
            grouped.setdefault(key, []).append(meta)
        records = [self._build_record(key, metas) for key, metas in grouped.items()]
        return sorted(records, key=lambda record: record.doc_id)

    def _document_record(self, doc_id: str) -> DocumentRecord | None:
        if self._collection is None:
            return None
        result = self._collection.get(where={"document": doc_id}, include=["metadatas"])
        metas = result.get("metadatas") or []
        if not metas:
            return None
        return self._build_record(doc_id, metas)

    @staticmethod
    def _build_record(doc_id: str, metas: list[dict[str, Any]]) -> DocumentRecord:
        child_ids: list[str] = []
        seen: set[str] = set()
        pages: set[int] = set()
        content_hash: str | None = None
        source: str | None = None
        modality: str | None = None
        for meta in metas:
            ref = meta.get("ref_doc_id") or meta.get("doc_id")
            if ref and ref not in seen:
                seen.add(ref)
                child_ids.append(ref)
            page = meta.get("page")
            if isinstance(page, int):
                pages.add(page)
            content_hash = content_hash or meta.get("content_hash")
            source = source or meta.get("source")
            modality = modality or meta.get("modality")
        return DocumentRecord(
            doc_id=doc_id,
            doc_ids=sorted(child_ids),
            source=source,
            modality=modality,
            pages=sorted(pages),
            chunks=len(metas),
            content_hash=content_hash,
        )

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
        """Responde a uma mensagem usando recuperação + Gemini.

        Retorna a resposta e as fontes (chunks) recuperadas.
        """
        chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            llm=self._llm,
            similarity_top_k=top_k or self._settings.similarity_top_k,
            system_prompt=CITATION_SYSTEM_PROMPT,
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
    def _delete_by_document(self, doc_id: str) -> bool:
        """Remove do Chroma todos os nós cujo metadado ``document`` é ``doc_id``.

        Cobre o documento inteiro (todas as páginas/abas) em uma operação, ao
        contrário de ``delete_ref_doc``, que só apagaria o ``doc_id`` exato e
        deixaria páginas derivadas (``doc_id:p2``) órfãs.
        """
        if self._collection is None:
            return False
        existing = self._collection.get(where={"document": doc_id})
        ids = existing.get("ids") or []
        if not ids:
            return False
        self._collection.delete(ids=ids)
        return True

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
