"""Testes da resolução de device (cuda/rocm/mps/cpu) — com ``torch`` falso.

Não dependem de PyTorch instalado nem de GPU: injetam um módulo ``torch`` falso
em ``sys.modules`` para exercitar cada ramo da detecção.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from app.core.devices import describe_device, resolve_device, resolve_torch_dtype


def _fake_torch(*, cuda: bool = False, mps: bool = False, hip: str | None = None) -> ModuleType:
    torch = ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.cuda = SimpleNamespace(is_available=lambda: cuda)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    torch.version = SimpleNamespace(cuda="12.1", hip=hip)
    return torch


def _install(monkeypatch, torch: ModuleType) -> None:
    monkeypatch.setitem(sys.modules, "torch", torch)


# --------------------------------------------------------------------------- #
# resolve_device
# --------------------------------------------------------------------------- #
def test_auto_prefers_cuda(monkeypatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=True, mps=True))
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_mps(monkeypatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=False, mps=True))
    assert resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu(monkeypatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=False, mps=False))
    assert resolve_device("auto") == "cpu"


def test_rocm_is_alias_for_cuda() -> None:
    # Não toca em torch: AMD ROCm é acessada pelo backend cuda do PyTorch.
    assert resolve_device("rocm") == "cuda"


@pytest.mark.parametrize("explicit", ["cuda", "mps", "cpu"])
def test_explicit_device_passes_through(explicit: str) -> None:
    assert resolve_device(explicit) == explicit


def test_invalid_device_raises() -> None:
    with pytest.raises(ValueError):
        resolve_device("gpu")


# --------------------------------------------------------------------------- #
# describe_device / resolve_torch_dtype
# --------------------------------------------------------------------------- #
def test_describe_device_non_cuda_is_verbatim() -> None:
    assert describe_device("cpu") == "cpu"


def test_describe_device_detects_amd_rocm(monkeypatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=True, hip="6.2"))
    assert describe_device("cuda") == "cuda (AMD ROCm)"


def test_describe_device_detects_nvidia(monkeypatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=True, hip=None))
    assert describe_device("cuda") == "cuda (NVIDIA)"


def test_torch_dtype_float16_on_cuda(monkeypatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=True))
    assert resolve_torch_dtype("cuda") == "float16"


def test_torch_dtype_float32_off_cuda(monkeypatch) -> None:
    _install(monkeypatch, _fake_torch(cuda=False))
    assert resolve_torch_dtype("cpu") == "float32"
