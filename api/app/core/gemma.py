"""Engine compartilhada do ``google/gemma-4-E2B-it``.

O Gemma 4 E2B é multimodal (texto e imagem). Para não carregar o modelo duas
vezes na memória, esta engine é a **única** dona do processor + modelo e é
reutilizada tanto pelo wrapper LLM do LlamaIndex (`GemmaLLM`) quanto pela camada
de ingestão (OCR/descrição de imagens e páginas de PDF). A transcrição de áudio
fica a cargo da :class:`~app.core.transcription.WhisperEngine`.

Referência de uso (model card, maio/2026):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                           return_tensors="pt", add_generation_prompt=True)
    outputs = model.generate(**inputs, max_new_tokens=...)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    processor.parse_response(response)
"""

from __future__ import annotations

from typing import Any

from app.config import Settings

# Parâmetros de amostragem recomendados pelo model card do Gemma 4.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64

# Mensagem = lista de dicts no formato de chat do Transformers.
Message = dict[str, Any]


def _resolve_dtype(dtype: str) -> Any:
    """Converte a string de config em um ``torch.dtype`` (ou a string "auto")."""
    if dtype == "auto":
        return "auto"
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"dtype inválido: {dtype!r}")
    return mapping[dtype]


class GemmaEngine:
    """Carrega e serve o Gemma 4 E2B para geração de texto e tarefas multimodais."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._processor: Any = None
        self._model: Any = None
        self._device: str = "cpu"

    # ------------------------------------------------------------------ #
    # Carregamento
    # ------------------------------------------------------------------ #
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Baixa (se necessário) e carrega o processor + modelo na memória."""
        if self.is_loaded:
            return

        from transformers import AutoModelForMultimodalLM, AutoProcessor

        s = self._settings
        token = s.hf_token or None
        common = {"cache_dir": s.hf_cache_dir, "token": token}

        self._processor = AutoProcessor.from_pretrained(s.llm_model, **common)

        device = self._resolve_device(s.device)

        model_kwargs: dict[str, Any] = {
            "dtype": _resolve_dtype(s.dtype),
            "low_cpu_mem_usage": True,
            **common,
        }
        # Em CUDA usamos o sharding do accelerate (device_map="auto"). Em CPU/MPS,
        # carregamos o modelo inteiro e o movemos para o device: device_map="auto"
        # sem GPU pode deixar tensores no device 'meta' e quebrar na geração
        # ("Tensor on device meta is not on the expected device cpu!").
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForMultimodalLM.from_pretrained(s.llm_model, **model_kwargs)
        self._model.eval()

        if device != "cuda":
            self._model = self._model.to(device)
        self._device = device

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve "auto" para um device concreto (cuda > mps > cpu)."""
        if device != "auto":
            return device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------ #
    # Geração
    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """Gera uma resposta de texto para uma conversa multimodal.

        ``messages`` segue o formato de chat do Transformers, podendo conter
        partes de texto, imagem e áudio.
        """
        self.load()
        s = self._settings
        max_new_tokens = max_new_tokens or s.max_new_tokens
        enable_thinking = s.enable_thinking if enable_thinking is None else enable_thinking

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(self._device)

        input_len = inputs["input_ids"].shape[-1]

        import torch

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                top_k=DEFAULT_TOP_K,
            )

        response = self._processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        return self._extract_text(self._processor.parse_response(response))

    # ------------------------------------------------------------------ #
    # Helpers multimodais (usados pela camada de ingestão)
    # ------------------------------------------------------------------ #
    def describe_image(self, image_path: str, *, prompt: str | None = None) -> str:
        """Extrai/descreve o conteúdo de uma imagem (OCR + descrição) via Gemma 4."""
        text = prompt or (
            "Extract and transcribe all text visible in this image. "
            "Then, in a new paragraph, briefly describe the visual content."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": text},
                ],
            }
        ]
        return self.generate(messages, enable_thinking=False)

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_text(parsed: Any) -> str:
        """Normaliza o retorno de ``processor.parse_response`` para texto puro.

        Dependendo da versão do Transformers, ``parse_response`` pode retornar
        uma string, um dict ``{"content": ...}`` ou um objeto com ``.content``.
        """
        if parsed is None:
            return ""
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict):
            for key in ("content", "text", "answer"):
                if key in parsed and isinstance(parsed[key], str):
                    return parsed[key].strip()
        content = getattr(parsed, "content", None)
        if isinstance(content, str):
            return content.strip()
        return str(parsed).strip()
