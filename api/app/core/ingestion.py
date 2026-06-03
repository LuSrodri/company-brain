"""Ingestão multimodal: transforma conteúdo bruto em ``Document`` do LlamaIndex.

Roteamento por tipo de arquivo (ferramentas usadas em cada caso):
    * .txt/.csv/.md/...   -> leitura direta do texto (1 documento, page=1)
    * .pdf                -> pdf2image rasteriza cada página e o Gemma 4 faz a
                             leitura (OCR + descrição); 1 documento POR PÁGINA,
                             com ``metadata["page"]``
    * imagem (.png/...)   -> OCR + descrição via Gemma 4 (visão), page=1
    * áudio (.mp3/...)    -> transcrição (STT) via Whisper, com timestamps
    * .xlsx/.xlsm         -> pandas + openpyxl (texto das planilhas) e Gemma 4
                             para descrever imagens embutidas; 1 documento por aba
    * .docx               -> MarkItDown (texto) + python-docx (imagens embutidas
                             descritas pelo Gemma 4); 1 documento, page=1
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from llama_index.core import Document

from app.core.transcription import WhisperEngine
from app.core.vision import ImageDescriber

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".csv", ".json"}
PDF_EXTS = {".pdf"}
XLSX_EXTS = {".xlsx", ".xlsm"}
DOCX_EXTS = {".docx"}

SUPPORTED_EXTS = IMAGE_EXTS | AUDIO_EXTS | TEXT_EXTS | PDF_EXTS | XLSX_EXTS | DOCX_EXTS

# Prompt de OCR para páginas de PDF e imagens de documentos: pede transcrição
# fiel, sem resumir/traduzir, preservando a ordem do conteúdo.
OCR_PROMPT = (
    "Transcribe ALL the text content of this document page verbatim, preserving "
    "the original language, wording, order and line breaks. Do not summarize, "
    "translate or add commentary. If there are figures, tables or images, append "
    "a short description of them after the transcribed text."
)
EMBEDDED_IMAGE_PROMPT = (
    "Transcribe any text visible in this image and then briefly describe its "
    "visual content. Keep the original language."
)


class UnsupportedFileTypeError(ValueError):
    """Levantado quando a extensão do arquivo não é suportada."""


class EmptyDocumentError(ValueError):
    """Levantado quando nenhum texto pôde ser extraído do arquivo."""


# Metadados de bookkeeping: ficam gravados no nó (para listar/excluir/atualizar),
# mas NÃO devem entrar no texto embeddado nem no contexto do LLM — são ruído sem
# valor semântico (ex.: o SHA-256 do conteúdo).
_NON_SEMANTIC_META = ["content_hash"]


def build_text_document(
    text: str, *, doc_id: str, metadata: dict[str, Any] | None = None
) -> Document:
    """Cria um ``Document`` a partir de texto puro, com ``doc_id`` estável.

    Chaves de bookkeeping (``content_hash``) são mantidas nos metadados, porém
    excluídas do texto usado para embedding e para o LLM (evita poluir o vetor
    com o hash).
    """
    return Document(
        text=text,
        doc_id=doc_id,
        metadata=metadata or {},
        excluded_embed_metadata_keys=_NON_SEMANTIC_META,
        excluded_llm_metadata_keys=_NON_SEMANTIC_META,
    )


def build_documents_from_file(
    path: str | Path,
    *,
    engine: ImageDescriber,
    stt_engine: WhisperEngine,
    doc_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Lê um arquivo do disco e devolve um ou mais ``Document`` com o texto extraído.

    PDFs geram um ``Document`` por página e planilhas um por aba (cada um com
    ``metadata["page"]``); os demais formatos geram um único documento (``page=1``).
    """
    path = Path(path)
    ext = path.suffix.lower()
    base_meta: dict[str, Any] = {
        **(metadata or {}),
        "source": path.name,
        "document": doc_id,
        "modality": _modality(ext),
    }

    if ext in PDF_EXTS:
        return _pdf_documents(path, engine=engine, doc_id=doc_id, base_meta=base_meta)
    if ext in XLSX_EXTS:
        return _xlsx_documents(path, engine=engine, doc_id=doc_id, base_meta=base_meta)
    if ext in DOCX_EXTS:
        return _docx_documents(path, engine=engine, doc_id=doc_id, base_meta=base_meta)

    if ext in TEXT_EXTS:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext in IMAGE_EXTS:
        text = engine.describe_image(str(path))
    elif ext in AUDIO_EXTS:
        text = stt_engine.transcribe(str(path))
    else:
        raise UnsupportedFileTypeError(
            f"Extensão não suportada: {ext!r}. Suportadas: {sorted(SUPPORTED_EXTS)}"
        )

    text = (text or "").strip()
    if not text:
        raise EmptyDocumentError(f"Nenhum texto extraído de {path.name}.")
    return [build_text_document(text, doc_id=doc_id, metadata={**base_meta, "page": 1})]


# --------------------------------------------------------------------------- #
# PDF: pdf2image (rasteriza) + Gemma 4 (OCR/descrição), 1 documento por página
# --------------------------------------------------------------------------- #
def _pdf_documents(
    path: Path, *, engine: ImageDescriber, doc_id: str, base_meta: dict[str, Any]
) -> list[Document]:
    """Rasteriza cada página do PDF e usa o Gemma 4 para extrair o texto (OCR)."""
    from app.config import get_settings
    from pdf2image import convert_from_path

    s = get_settings()
    images = convert_from_path(
        str(path),
        dpi=s.pdf_dpi,
        poppler_path=s.poppler_path or None,
    )

    documents: list[Document] = []
    for page_number, image in enumerate(images, start=1):
        text = _describe_pil_image(image, engine=engine, prompt=OCR_PROMPT)
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
        raise EmptyDocumentError(f"Nenhum texto extraído do PDF {path.name}.")
    return documents


# --------------------------------------------------------------------------- #
# XLSX: pandas (texto das abas) + openpyxl (imagens embutidas -> Gemma 4)
# --------------------------------------------------------------------------- #
def _xlsx_documents(
    path: Path, *, engine: ImageDescriber, doc_id: str, base_meta: dict[str, Any]
) -> list[Document]:
    """Extrai o texto de cada aba (pandas) e descreve imagens embutidas (openpyxl)."""
    import pandas as pd

    sheets: dict[str, pd.DataFrame] = pd.read_excel(
        path, sheet_name=None, header=None, engine="openpyxl"
    )
    images_by_sheet = _xlsx_images(path, engine=engine)

    documents: list[Document] = []
    for index, (sheet_name, frame) in enumerate(sheets.items(), start=1):
        parts = [_dataframe_to_text(sheet_name, frame)]
        parts.extend(images_by_sheet.get(sheet_name, []))
        text = "\n\n".join(p for p in parts if p).strip()
        if not text:
            continue
        documents.append(
            build_text_document(
                text,
                doc_id=f"{doc_id}:s{index}",
                metadata={**base_meta, "page": index, "sheet": sheet_name},
            )
        )

    if not documents:
        raise EmptyDocumentError(f"Nenhum conteúdo extraído da planilha {path.name}.")
    return documents


def _dataframe_to_text(sheet_name: str, frame: Any) -> str:
    """Serializa uma aba (DataFrame) em linhas legíveis ``v | v | v``."""
    frame = frame.fillna("")
    lines = [
        " | ".join(str(value) for value in row).strip(" |")
        for row in frame.values.tolist()
    ]
    body = "\n".join(line for line in lines if line)
    return f"Planilha: {sheet_name}\n{body}".strip()


def _xlsx_images(path: Path, *, engine: ImageDescriber) -> dict[str, list[str]]:
    """Descreve, via Gemma 4, imagens embutidas em cada aba do workbook."""
    descriptions: dict[str, list[str]] = {}
    try:
        from openpyxl import load_workbook

        workbook = load_workbook(path)
    except Exception:  # noqa: BLE001 — workbook sem imagens / formato inesperado
        return descriptions

    for worksheet in workbook.worksheets:
        for image in getattr(worksheet, "_images", []):
            data = _openpyxl_image_bytes(image)
            if not data:
                continue
            described = _describe_image_bytes(data, engine=engine, prompt=EMBEDDED_IMAGE_PROMPT)
            if described:
                descriptions.setdefault(worksheet.title, []).append(described)
    return descriptions


def _openpyxl_image_bytes(image: Any) -> bytes | None:
    """Extrai os bytes brutos de uma imagem carregada pelo openpyxl."""
    try:
        data_fn = getattr(image, "_data", None)
        if callable(data_fn):
            return data_fn()
        ref = getattr(image, "ref", None)
        if hasattr(ref, "read"):
            ref.seek(0)
            return ref.read()
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------- #
# DOCX: MarkItDown (texto) + python-docx (imagens embutidas -> Gemma 4)
# --------------------------------------------------------------------------- #
def _docx_documents(
    path: Path, *, engine: ImageDescriber, doc_id: str, base_meta: dict[str, Any]
) -> list[Document]:
    """Converte o .docx em Markdown (MarkItDown) e descreve imagens (python-docx)."""
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    text = (getattr(result, "text_content", None) or "").strip()

    parts = [text, *_docx_image_descriptions(path, engine=engine)]
    full_text = "\n\n".join(p for p in parts if p).strip()
    if not full_text:
        raise EmptyDocumentError(f"Nenhum conteúdo extraído do documento {path.name}.")
    return [build_text_document(full_text, doc_id=doc_id, metadata={**base_meta, "page": 1})]


def _docx_image_descriptions(path: Path, *, engine: ImageDescriber) -> list[str]:
    """Descreve, via Gemma 4, as imagens embutidas em um .docx."""
    descriptions: list[str] = []
    try:
        from docx import Document as DocxDocument

        document = DocxDocument(str(path))
    except Exception:  # noqa: BLE001
        return descriptions

    for rel in document.part.rels.values():
        if "image" not in rel.reltype:
            continue
        try:
            blob = rel.target_part.blob
        except Exception:  # noqa: BLE001
            continue
        described = _describe_image_bytes(blob, engine=engine, prompt=EMBEDDED_IMAGE_PROMPT)
        if described:
            descriptions.append(described)
    return descriptions


# --------------------------------------------------------------------------- #
# Helpers de imagem (gravam um arquivo temporário e chamam o Gemma 4)
# --------------------------------------------------------------------------- #
def _describe_pil_image(image: Any, *, engine: ImageDescriber, prompt: str) -> str:
    """Salva uma imagem PIL em arquivo temporário e a descreve via Gemma 4."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        image.save(tmp_path, format="PNG")
        return (engine.describe_image(str(tmp_path), prompt=prompt) or "").strip()
    finally:
        tmp_path.unlink(missing_ok=True)


def _describe_image_bytes(data: bytes, *, engine: ImageDescriber, prompt: str) -> str:
    """Grava bytes de imagem em arquivo temporário e os descreve via Gemma 4."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        return (engine.describe_image(str(tmp_path), prompt=prompt) or "").strip()
    finally:
        tmp_path.unlink(missing_ok=True)


def _modality(ext: str) -> str:
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in PDF_EXTS:
        return "pdf"
    if ext in XLSX_EXTS:
        return "spreadsheet"
    if ext in DOCX_EXTS:
        return "document"
    return "text"
