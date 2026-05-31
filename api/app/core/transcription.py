"""Engine de transcrição de áudio (STT) servida pelo Whisper via Hugging Face.

Usa ``openai/whisper-large-v3-turbo`` (multilíngue, 99 idiomas incluindo PT-BR)
pela pipeline ``automatic-speech-recognition`` do Transformers, rodando
localmente na GPU/CPU (o device é resolvido por :mod:`app.core.devices`).

A transcrição preserva **timestamps por segmento**: o texto resultante é uma
sequência de linhas no formato ``[HH:MM:SS] trecho transcrito``, de modo que a
recuperação (RAG) consegue devolver, junto do parágrafo, o instante do áudio em
que aquele trecho foi falado.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.core.devices import resolve_device, resolve_torch_dtype

# O Whisper opera com áudio reamostrado a 16 kHz.
TARGET_SAMPLE_RATE = 16_000


def _format_timestamp(seconds: float | None) -> str:
    """Formata um instante (em segundos) como ``HH:MM:SS``."""
    total = int(round(seconds or 0.0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class WhisperEngine:
    """Carrega e serve o Whisper para transcrição de áudio com timestamps."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: Any = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        """Baixa (se necessário) e instancia a pipeline de ASR na memória."""
        if self.is_loaded:
            return

        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        s = self._settings
        device = resolve_device(s.device)
        torch_dtype = resolve_torch_dtype(device)

        common = {"cache_dir": s.hf_cache_dir, "token": s.hf_token or None}
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            s.stt_model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **common,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(s.stt_model, **common)

        pipeline_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
            "torch_dtype": torch_dtype,
            "device": device,
        }
        # Só ativa a janela deslizante se explicitamente configurada (>0). Por
        # padrão usamos o long-form nativo do Whisper, que dá timestamps precisos
        # por segmento (a janela deslizante do transformers colapsa os segmentos).
        if s.stt_chunk_length_s and s.stt_chunk_length_s > 0:
            pipeline_kwargs["chunk_length_s"] = s.stt_chunk_length_s

        self._pipeline = pipeline("automatic-speech-recognition", **pipeline_kwargs)

    # ------------------------------------------------------------------ #
    # Transcrição
    # ------------------------------------------------------------------ #
    def transcribe(self, audio_path: str) -> str:
        """Transcreve um arquivo de áudio em texto com timestamps por segmento.

        Retorna uma string multi-linha (uma linha ``[HH:MM:SS] texto`` por
        segmento). Áudios longos são tratados pelo long-form nativo do Whisper
        (ou pela janela deslizante, se ``CB_STT_CHUNK_LENGTH_S`` > 0); o idioma é
        autodetectado, salvo se ``CB_STT_LANGUAGE`` força um.
        """
        self.load()

        # Decodifica o áudio para um array mono 16 kHz com librosa, evitando a
        # dependência externa do ffmpeg na pipeline (mp3/m4a/flac/etc.).
        import librosa

        waveform, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)

        call_kwargs: dict[str, Any] = {"return_timestamps": True}
        if self._settings.stt_language:
            call_kwargs["generate_kwargs"] = {"language": self._settings.stt_language}

        result = self._pipeline(
            {"raw": waveform, "sampling_rate": TARGET_SAMPLE_RATE},
            **call_kwargs,
        )
        return self._format_result(result)

    @staticmethod
    def _format_result(result: dict[str, Any]) -> str:
        """Converte a saída da pipeline em ``[HH:MM:SS] texto`` por segmento."""
        chunks = result.get("chunks") if isinstance(result, dict) else None
        if not chunks:
            # Sem timestamps por segmento: devolve o texto puro.
            text = (result.get("text") if isinstance(result, dict) else None) or ""
            return text.strip()

        lines: list[str] = []
        for chunk in chunks:
            start = chunk.get("timestamp", (None, None))[0]
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            lines.append(f"[{_format_timestamp(start)}] {text}")
        return "\n".join(lines).strip()
