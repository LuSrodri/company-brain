"""Descrição/OCR de imagens via o LLM multimodal (gemini-3.1-flash-lite).

Usa o **mesmo LLM oficial do LlamaIndex** (`llama-index-llms-google-genai`,
:class:`GoogleGenAI`) que serve o chat, agora também para descrever imagens na
camada de ingestão (páginas de PDF rasterizadas, imagens soltas e imagens
embutidas em planilhas/documentos).

A descrição é feita com uma única mensagem de usuário multimodal
(``ChatMessage`` com ``ImageBlock`` + ``TextBlock``).

Resiliência a *rate limit*: a ingestão de um PDF dispara uma chamada de visão
**por página**, em rajada — fácil de estourar o limite por minuto (RPM) da API e
receber ``429 RESOURCE_EXHAUSTED``. Por isso o ``describe_image`` repete com
*backoff* nesses casos (respeitando o ``retryDelay`` da resposta quando presente).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Prompt padrão (OCR + descrição) para imagens sem prompt específico.
DEFAULT_IMAGE_PROMPT = (
    "Extract and transcribe all text visible in this image. "
    "Then, in a new paragraph, briefly describe the visual content."
)

# Retry em 429 (rate limit por minuto). Backoff exponencial limitado: a janela
# de RPM costuma resetar em ~60s, então alguns minutos no pior caso bastam.
_RATE_LIMIT_MAX_RETRIES = 5
_RATE_LIMIT_BASE_DELAY_S = 8.0
_RATE_LIMIT_MAX_DELAY_S = 60.0


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Usa o ``retryDelay`` da resposta (ex.: ``'retryDelay': '20s'``) se houver;
    senão, backoff exponencial limitado."""
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    if match:
        return min(float(match.group(1)), _RATE_LIMIT_MAX_DELAY_S)
    return min(_RATE_LIMIT_BASE_DELAY_S * (2**attempt), _RATE_LIMIT_MAX_DELAY_S)


class ImageDescriber:
    """Descreve imagens (OCR + descrição) usando o LLM ``GoogleGenAI`` do chat."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def describe_image(self, image_path: str, *, prompt: str | None = None) -> str:
        """Extrai/descreve o conteúdo de uma imagem do disco via o LLM multimodal.

        Repete com *backoff* em ``429`` (rate limit); demais erros sobem na hora
        (o ``GoogleGenAI`` já trata 500/503 internamente).
        """
        from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock

        message = ChatMessage(
            role="user",
            blocks=[
                ImageBlock(path=image_path),
                TextBlock(text=prompt or DEFAULT_IMAGE_PROMPT),
            ],
        )

        for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            try:
                response = self._llm.chat([message])
                return (str(response.message.content) or "").strip()
            except Exception as exc:  # noqa: BLE001 — re-levanta se não for rate limit
                if not _is_rate_limit_error(exc) or attempt == _RATE_LIMIT_MAX_RETRIES:
                    raise
                delay = _retry_delay_seconds(exc, attempt)
                logger.warning(
                    "429 (rate limit) ao descrever imagem; retry %d/%d em %.0fs",
                    attempt + 1,
                    _RATE_LIMIT_MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("describe_image: laço de retry terminou sem retornar")  # pragma: no cover
