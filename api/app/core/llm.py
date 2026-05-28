"""Adaptador do Gemma 4 E2B para a interface ``LLM`` do LlamaIndex.

Encapsula uma :class:`GemmaEngine` (que detém o modelo) e a expõe como um
``CustomLLM`` que o LlamaIndex usa nos query/chat engines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from pydantic import Field, PrivateAttr

from app.config import Settings
from app.core.gemma import GemmaEngine

# Janela de contexto do Gemma 4 E2B (128K tokens).
GEMMA_E2B_CONTEXT_WINDOW = 128_000


class GemmaLLM(CustomLLM):
    """LLM do LlamaIndex servido pelo Gemma 4 E2B via :class:`GemmaEngine`."""

    model_name: str = Field(default="google/gemma-4-E2B-it")
    max_new_tokens: int = Field(default=1024)
    enable_thinking: bool = Field(default=False)
    context_window: int = Field(default=GEMMA_E2B_CONTEXT_WINDOW)

    _engine: GemmaEngine = PrivateAttr()

    def __init__(self, engine: GemmaEngine, settings: Settings, **kwargs: Any) -> None:
        super().__init__(
            model_name=settings.llm_model,
            max_new_tokens=settings.max_new_tokens,
            enable_thinking=settings.enable_thinking,
            **kwargs,
        )
        self._engine = engine

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=True,
        )

    # ------------------------------------------------------------------ #
    # Chat (caminho preferencial — usa o chat template nativo do Gemma 4)
    # ------------------------------------------------------------------ #
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        text = self._engine.generate(
            self._to_engine_messages(messages),
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            enable_thinking=kwargs.get("enable_thinking", self.enable_thinking),
        )
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        # Geração não-streaming envolvida como um gerador de um único yield.
        response = self.chat(messages, **kwargs)

        def gen() -> ChatResponseGen:
            yield response

        return gen()

    # ------------------------------------------------------------------ #
    # Completion (mapeado para uma única mensagem de usuário)
    # ------------------------------------------------------------------ #
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        text = self._engine.generate(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            enable_thinking=kwargs.get("enable_thinking", self.enable_thinking),
        )
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        response = self.complete(prompt, formatted=formatted, **kwargs)

        def gen() -> CompletionResponseGen:
            yield response

        return gen()

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_engine_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
        """Converte ``ChatMessage`` do LlamaIndex no formato de chat do Transformers."""
        converted: list[dict[str, Any]] = []
        for msg in messages:
            converted.append(
                {
                    "role": msg.role.value,
                    "content": [{"type": "text", "text": msg.content or ""}],
                }
            )
        return converted
