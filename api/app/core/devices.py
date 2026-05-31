"""Resolução de dispositivo (device) compartilhada pelas engines locais.

Após a migração do LLM para o Google AI Studio, quem ainda roda localmente via
PyTorch/Hugging Face é a transcrição (:class:`~app.core.transcription.WhisperEngine`)
e os embeddings (:func:`~app.core.embeddings.build_embed_model`). Este módulo
centraliza a escolha de device — antes duplicada em cada engine — cobrindo:

    * **NVIDIA**  -> CUDA;
    * **AMD**     -> ROCm, que no PyTorch também é exposto como ``cuda``
                     (``torch.version.hip`` indica a build HIP/ROCm);
    * **Apple**   -> MPS;
    * **fallback**-> CPU.

Não há suporte a DirectML: o ``torch-directml`` prende o PyTorch em 2.3.1, o que
quebraria ``transformers``/``sentence-transformers`` atuais. Em GPUs AMD não
cobertas pelo ROCm (ex.: RX 580 / Polaris), a detecção cai para CPU — esperado.
"""

from __future__ import annotations

from typing import Any

# Ordem de preferência ao resolver ``device="auto"``.
_VALID_DEVICES = {"auto", "cuda", "rocm", "mps", "cpu"}


def resolve_device(setting: str) -> str:
    """Resolve a configuração de device em um device concreto do PyTorch.

    ``"auto"`` escolhe ``cuda`` (NVIDIA ou AMD ROCm) > ``mps`` > ``cpu``. Valores
    explícitos são respeitados; ``"rocm"`` é um alias para ``"cuda"`` (no PyTorch,
    GPUs AMD com ROCm são acessadas pelo backend ``cuda``).
    """
    if setting not in _VALID_DEVICES:
        raise ValueError(
            f"device inválido: {setting!r}. Use um de {sorted(_VALID_DEVICES)}."
        )

    if setting == "rocm":
        return "cuda"
    if setting != "auto":
        return setting

    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    """Rótulo legível do device para logs (distingue NVIDIA CUDA de AMD ROCm)."""
    if device != "cuda":
        return device

    import torch

    # Builds ROCm do PyTorch expõem ``torch.version.hip``; CUDA expõe só ``version.cuda``.
    if getattr(torch.version, "hip", None):
        return "cuda (AMD ROCm)"
    return "cuda (NVIDIA)"


def resolve_torch_dtype(device: str) -> Any:
    """``torch.dtype`` adequado ao device: ``float16`` em GPU CUDA, senão ``float32``."""
    import torch

    return torch.float16 if device == "cuda" else torch.float32
