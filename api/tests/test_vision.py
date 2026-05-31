"""Testes do :class:`ImageDescriber` — com um LLM falso (sem rede).

Verifica que a descrição de imagem monta uma ``ChatMessage`` multimodal
(``ImageBlock`` apontando para o arquivo + ``TextBlock`` com o prompt) e devolve
o conteúdo textual da resposta do LLM.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llama_index.core.llms import ImageBlock, TextBlock

from app.core.vision import DEFAULT_IMAGE_PROMPT, ImageDescriber


class FakeLLM:
    """LLM falso: registra as mensagens recebidas e devolve um texto fixo."""

    def __init__(self, reply: str = "TEXTO OCR + descrição") -> None:
        self.reply = reply
        self.calls: list = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append(messages)
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
