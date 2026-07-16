"""Heurísticas de métrica e o LLM-as-judge da avaliação.

As checagens determinísticas (recall de recuperação, citação, recusa,
resistência a injeção) usam normalização robusta de texto. A **correção** da
resposta é avaliada por um segundo LLM (o próprio Gemini) atuando como juiz,
retornando um veredito binário + justificativa.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Any

# Marcadores de recusa: a resposta admite não ter a informação na base.
_REFUSAL_MARKERS = [
    "nao encontr",
    "nao consta",
    "nao ha informac",
    "nao foi encontr",
    "nao localiz",
    "nao tenho essa informac",
    "nao esta na base",
    "not found",
    "no information",
    "does not contain",
    "unable to find",
    "cannot find",
]


def normalize(text: str) -> str:
    """Minúsculas, sem acentos, espaços colapsados — para comparação robusta."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def retrieval_hit(expected: list[str], sources_text: str) -> bool:
    """True se TODOS os trechos esperados aparecem no texto das fontes recuperadas."""
    norm_sources = normalize(sources_text)
    return all(normalize(chunk) in norm_sources for chunk in expected)


def citation_present(answer: str, citation_hint: str) -> bool:
    """Heurística: a resposta cita a fonte (nome do arquivo, página ou timestamp)."""
    norm_answer = normalize(answer)
    # Nome do arquivo (sem extensão) mencionado na resposta.
    stem = normalize(re.sub(r"\.[a-z0-9]+$", "", citation_hint))
    if stem and stem in norm_answer:
        return True
    # Timestamp [HH:MM:SS] ou HH:MM:SS.
    if re.search(r"\d{2}:\d{2}:\d{2}", answer):
        return True
    # Referência de página: "p. 3", "pagina 3", "page 3".
    if re.search(r"\b(p\.?|pag\.?|pagina|page)\s*\d+", norm_answer):
        return True
    return False


def is_refusal(answer: str) -> bool:
    """True se a resposta admite não ter a informação (recusa correta)."""
    norm_answer = normalize(answer)
    return any(marker in norm_answer for marker in _REFUSAL_MARKERS)


def percentiles(values: list[float]) -> dict[str, float]:
    """p50/p95/média/máximo de uma lista de latências (ms)."""
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))],
        "mean": statistics.fmean(ordered),
        "max": ordered[-1],
        "n": len(ordered),
    }


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = """Você é um avaliador rigoroso de respostas de um sistema de \
RAG corporativo. Dada a PERGUNTA, a INFORMAÇÃO DE REFERÊNCIA (que a resposta \
correta deve refletir) e a RESPOSTA do sistema, decida se a resposta está \
factualmente correta e responde à pergunta com base na referência.

Responda APENAS com um JSON válido, sem texto ao redor, no formato:
{{"correct": true|false, "reason": "<justificativa curta>"}}

PERGUNTA:
{question}

INFORMAÇÃO DE REFERÊNCIA:
{reference}

RESPOSTA DO SISTEMA:
{answer}
"""


@dataclass
class JudgeVerdict:
    correct: bool
    reason: str


class LLMJudge:
    """Juiz baseado no Gemini (Google AI Studio). ``complete`` retorna texto."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def judge(self, *, question: str, reference: str, answer: str) -> JudgeVerdict:
        prompt = _JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
        raw = str(self._llm.complete(prompt))
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> JudgeVerdict:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return JudgeVerdict(correct=False, reason=f"veredito não-JSON: {raw[:120]!r}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return JudgeVerdict(correct=False, reason=f"JSON inválido: {raw[:120]!r}")
        return JudgeVerdict(
            correct=bool(data.get("correct", False)),
            reason=str(data.get("reason", "")),
        )
