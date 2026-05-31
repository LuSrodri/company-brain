"""Descrição/OCR de imagens via o LLM multimodal (gemma-4-31b-it).

Substitui a antiga ``GemmaEngine``: em vez de um cliente ``google-genai`` próprio,
usa o **mesmo LLM oficial do LlamaIndex** (`llama-index-llms-google-genai`,
:class:`GoogleGenAI`) que serve o chat, agora também para descrever imagens na
camada de ingestão (páginas de PDF rasterizadas, imagens soltas e imagens
embutidas em planilhas/documentos).

A descrição é feita com uma única mensagem de usuário multimodal
(``ChatMessage`` com ``ImageBlock`` + ``TextBlock``); o ``GoogleGenAI`` cuida de
enviar a imagem (upload/inline) ao Gemma 4.
"""

from __future__ import annotations

from typing import Any

# Prompt padrão (OCR + descrição) para imagens sem prompt específico.
DEFAULT_IMAGE_PROMPT = (
    "Extract and transcribe all text visible in this image. "
    "Then, in a new paragraph, briefly describe the visual content."
)


class ImageDescriber:
    """Descreve imagens (OCR + descrição) usando o LLM ``GoogleGenAI`` do chat."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def describe_image(self, image_path: str, *, prompt: str | None = None) -> str:
        """Extrai/descreve o conteúdo de uma imagem do disco via Gemma 4 multimodal."""
        from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock

        message = ChatMessage(
            role="user",
            blocks=[
                ImageBlock(path=image_path),
                TextBlock(text=prompt or DEFAULT_IMAGE_PROMPT),
            ],
        )
        response = self._llm.chat([message])
        return (str(response.message.content) or "").strip()
