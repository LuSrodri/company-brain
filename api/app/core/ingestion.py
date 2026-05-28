"""Ingestão multimodal: transforma conteúdo bruto em ``Document`` do LlamaIndex.

Roteamento por tipo de arquivo:
    * texto (.txt/.md)      -> leitura direta
    * .pdf                  -> extração de texto com pypdf
    * imagem (.png/.jpg...) -> OCR + descrição via Gemma 4 (visão)
    * áudio (.wav/.mp3...)  -> transcrição (STT) via Gemma 4
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


def build_text_document(
    text: str, *, doc_id: str, metadata: dict[str, Any] | None = None
) -> Document:
    """Cria um ``Document`` a partir de texto puro, com ``doc_id`` estável."""
    return Document(text=text, doc_id=doc_id, metadata=metadata or {})


def build_document_from_file(
    path: str | Path,
    *,
    engine: GemmaEngine,
    doc_id: str,
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Lê um arquivo do disco e devolve um ``Document`` com o texto extraído."""
    path = Path(path)
    ext = path.suffix.lower()
    meta: dict[str, Any] = {"source": path.name, "modality": _modality(ext), **(metadata or {})}

    if ext in TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext in PDF_EXTS:
        text = _extract_pdf_text(path)
    elif ext in IMAGE_EXTS:
        text = engine.describe_image(str(path))
    elif ext in AUDIO_EXTS:
        text = engine.transcribe_audio(str(path))
    else:
        raise UnsupportedFileTypeError(
            f"Extensão não suportada: {ext!r}. Suportadas: {sorted(SUPPORTED_EXTS)}"
        )

    return build_text_document(text, doc_id=doc_id, metadata=meta)


def _modality(ext: str) -> str:
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in PDF_EXTS:
        return "pdf"
    return "text"


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()
