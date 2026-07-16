"""Schemas do endpoint de chat."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """Uma mensagem do histórico da conversa."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Mensagem atual do usuário.")
    history: list[ChatTurn] = Field(
        default_factory=list, description="Histórico anterior da conversa."
    )
    top_k: int | None = Field(
        default=None, ge=1, le=20, description="Sobrescreve o top_k de recuperação."
    )


class Source(BaseModel):
    text: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    latency_ms: float | None = Field(
        default=None, description="Latência server-side do chat (recuperação + geração), em ms."
    )
