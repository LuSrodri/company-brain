"""Testes do :class:`ImageDescriber` — com um LLM falso (sem rede).

Verifica que a descrição de imagem monta uma ``ChatMessage`` multimodal
(``ImageBlock`` apontando para o arquivo + ``TextBlock`` com o prompt) e devolve
o conteúdo textual da resposta do LLM.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.llms import ImageBlock, TextBlock

from app.core import vision
from app.core.vision import DEFAULT_IMAGE_PROMPT, ImageDescriber


class FakeLLM:
    """LLM falso: registra as mensagens recebidas e devolve um texto fixo."""

    def __init__(self, reply: str = "TEXTO OCR + descrição") -> None:
        self.reply = reply
        self.calls: list = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append(messages)
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


class FlakyLLM:
    """LLM falso que falha com 429 nas primeiras N chamadas e depois responde."""

    def __init__(self, fails: int, reply: str = "OK", exc_text: str | None = None) -> None:
        self.fails = fails
        self.reply = reply
        self.exc_text = exc_text or "ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429}}"
        self.calls = 0

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError(self.exc_text)
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


def _png(tmp_path: Path) -> Path:
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\nfake-bytes")
    return img


def test_describe_image_returns_text_and_builds_blocks(tmp_path: Path) -> None:
    img = _png(tmp_path)
    llm = FakeLLM()

    out = ImageDescriber(llm).describe_image(str(img), prompt="descreva isto")

    assert out == "TEXTO OCR + descrição"
    blocks = llm.calls[0][0].blocks
    image_blocks = [b for b in blocks if isinstance(b, ImageBlock)]
    text_blocks = [b for b in blocks if isinstance(b, TextBlock)]
    assert image_blocks and str(image_blocks[0].path) == str(img)
    assert text_blocks and text_blocks[0].text == "descreva isto"


def test_describe_image_uses_default_prompt(tmp_path: Path) -> None:
    img = _png(tmp_path)
    llm = FakeLLM()

    ImageDescriber(llm).describe_image(str(img))

    text_blocks = [b for b in llm.calls[0][0].blocks if isinstance(b, TextBlock)]
    assert text_blocks[0].text == DEFAULT_IMAGE_PROMPT


def test_describe_image_strips_whitespace(tmp_path: Path) -> None:
    img = _png(tmp_path)
    llm = FakeLLM(reply="   resultado com espaços   ")

    assert ImageDescriber(llm).describe_image(str(img)) == "resultado com espaços"


def test_describe_image_retries_on_rate_limit(tmp_path: Path, monkeypatch) -> None:
    """429 (rate limit) deve ser repetido com backoff até obter sucesso."""
    sleeps: list[float] = []
    monkeypatch.setattr(vision.time, "sleep", lambda s: sleeps.append(s))

    img = _png(tmp_path)
    llm = FlakyLLM(fails=2, reply="transcrição da página")

    out = ImageDescriber(llm).describe_image(str(img))

    assert out == "transcrição da página"
    assert llm.calls == 3  # 2 falhas + 1 sucesso
    assert len(sleeps) == 2  # esperou entre as tentativas


def test_describe_image_respects_retry_delay_hint(tmp_path: Path, monkeypatch) -> None:
    """Quando a resposta traz retryDelay, o backoff usa esse valor."""
    sleeps: list[float] = []
    monkeypatch.setattr(vision.time, "sleep", lambda s: sleeps.append(s))

    img = _png(tmp_path)
    llm = FlakyLLM(
        fails=1,
        exc_text="429 RESOURCE_EXHAUSTED {'error': {'code': 429, 'details': [{'retryDelay': '17s'}]}}",
    )

    ImageDescriber(llm).describe_image(str(img))

    assert sleeps == [17.0]


def test_describe_image_reraises_non_rate_limit_error(tmp_path: Path) -> None:
    """Erros que não são 429 sobem imediatamente, sem retry."""
    img = _png(tmp_path)
    llm = FlakyLLM(fails=1, exc_text="ValueError: imagem inválida")

    with pytest.raises(RuntimeError, match="imagem inválida"):
        ImageDescriber(llm).describe_image(str(img))
    assert llm.calls == 1
