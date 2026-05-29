"""Ingestão multimodal: transforma conteúdo bruto em ``Document`` do LlamaIndex.

Roteamento por tipo de arquivo:
    * texto (.txt/.md)      -> leitura direta (1 documento, page=1)
    * .pdf                  -> 1 documento POR PÁGINA, com metadata["page"]
                              extraída pelo pypdf
    * imagem (.png/.jpg...) -> OCR + descrição via Gemma 4 (visão), page=1
    * áudio (.wav/.mp3...)  -> transcrição (STT) via Gemma 4, page=1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llama_index.core import Document

from app.core.gemma import GemmaEngine

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".csv", ".json"}
PDF_EXTS = {".pdf"}

SUPPORTED_EXTS = IMAGE_EXTS | AUDIO_EXTS | TEXT_EXTS | PDF_EXTS


class UnsupportedFileTypeError(ValueError):
    """Levantado quando a extensão do arquivo não é suportada."""


class EmptyDocumentError(ValueError):
    """Levantado quando nenhum texto pôde ser extraído do arquivo."""


def build_text_document(
    text: str, *, doc_id: str, metadata: dict[str, Any] | None = None
) -> Document:
    """Cria um ``Document`` a partir de texto puro, com ``doc_id`` estável."""
    return Document(text=text, doc_id=doc_id, metadata=metadata or {})


def build_documents_from_file(
    path: str | Path,
    *,
    engine: GemmaEngine,
    doc_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Lê um arquivo do disco e devolve um ou mais ``Document`` com o texto extraído.

    PDFs geram um ``Document`` por página (com ``metadata["page"]``); os demais
    formatos geram um único documento (``page=1``).
    """
    path = Path(path)
    ext = path.suffix.lower()
    base_meta: dict[str, Any] = {
        "source": path.name,
        "document": doc_id,
        "modality": _modality(ext),
        **(metadata or {}),
    }

    if ext in PDF_EXTS:
        return _pdf_documents(path, doc_id=doc_id, base_meta=base_meta)

    if ext in TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext in IMAGE_EXTS:
        text = engine.describe_image(str(path))
    elif ext in AUDIO_EXTS:
        text = engine.transcribe_audio(str(path))
    else:
        raise UnsupportedFileTypeError(
            f"Extensão não suportada: {ext!r}. Suportadas: {sorted(SUPPORTED_EXTS)}"
        )

    text = (text or "").strip()
    if not text:
        raise EmptyDocumentError(f"Nenhum texto extraído de {path.name}.")
    return [build_text_document(text, doc_id=doc_id, metadata={**base_meta, "page": 1})]


def _pdf_documents(
    path: Path, *, doc_id: str, base_meta: dict[str, Any]
) -> list[Document]:
    """Extrai o texto de cada página do PDF, criando um ``Document`` por página."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(
            build_text_document(
                text,
                doc_id=f"{doc_id}:p{page_number}",
                metadata={**base_meta, "page": page_number},
            )
        )

    if not documents:
        raise EmptyDocumentError(
            f"Nenhum texto extraído do PDF {path.name} "
            "(pode ser um PDF escaneado/somente imagem)."
        )
    return documents


def _modality(ext: str) -> str:
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in PDF_EXTS:
        return "pdf"
    return "text"
