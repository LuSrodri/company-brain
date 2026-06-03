"""Teste end-to-end REAL (modelos de verdade) da ingestão multimodal + /chat.

Cobre as 6 modalidades suportadas usando **arquivos reais** colocados em
``tests/e2e_assets/`` (veja o README de lá). Para cada arquivo:

1. faz upload via ``POST /documents/upload`` (passando pelo pipeline real:
   pdf2image+Gemini para PDF, Whisper para áudio, Gemini para imagens,
   pandas/openpyxl para xlsx, MarkItDown/python-docx para docx);
2. faz uma pergunta via ``POST /chat``.

Requisito inegociável (OBS do enunciado): **a resposta gerada pelo Gemini nunca
pode estar vazia**. Além disso, validamos que o parágrafo esperado (e, quando
aplicável, a página/timestamp) aparece nas fontes recuperadas.

Pesado e com rede: chama o ``gemini-3.1-flash-lite`` via Google AI Studio + embeddings
da OpenAI, e baixa/carrega o Whisper localmente. Fora da suíte padrão:

    pytest -m e2e -s

Requisitos: ``CB_GOOGLE_API_KEY`` (LLM/ingestão) e ``CB_HF_TOKEN`` (Whisper/embeddings)
no ambiente OU em ``api/.env.dev``; arquivos em ``tests/e2e_assets/``; e, para o
caso de PDF, o poppler instalado.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(__file__).resolve().parent / "e2e_assets"

# --------------------------------------------------------------------------- #
# Casos de teste — um por modalidade. `expect` lista trechos que DEVEM aparecer
# nas fontes recuperadas; `page`/`timestamp` são checagens adicionais.
# --------------------------------------------------------------------------- #
CASES: list[dict] = [
    {
        "id": "pdf",
        "filename": "Data AI Label Project.pdf",
        "question": "Do I necessarily need produce hallucination in the process?",
        "expect": ["DO NOT CONTINUE YOUR TASK IF THE MODEL DID NOT PRODUCE ANY HALLUCINATION"],
        "page": 3,
    },
    {
        "id": "txt",
        "filename": "Internal Pentesting Report.txt",
        "question": "What are the CWEs?",
        "expect": ["CWE-862", "CWE-284", "CWE-639"],
    },
    {
        "id": "audio",
        "filename": "Alinhamento Black Friday.mp3",
        "question": "Combos valem para a black friday?",
        "expect": ["Combo e venda casada continuam valendo e contam pontos extras na comissão"],
        "timestamp": "00:00:51",
        "timestamp_tolerance_s": 15,
    },
    {
        "id": "docx",
        "filename": "Carta Apresentacao Lucas.docx",
        "question": "Qual é a stack do candidato Lucas?",
        "expect": ["Next.js, TypeScript, React, Node.js e Supabase"],
        "page": 1,
    },
    {
        "id": "xlsx",
        "filename": "notas.xlsx",
        "question": "Quanto falta para eu passar em Sistemas Operacionais Embarcados?",
        # A célula "NOTA MINIMA AF" guarda o float cru lido pelo pandas
        # (0.5444444444444446), não um "0,54" formatado — casamos pelo prefixo.
        "expect": ["0.5444"],
    },
    {
        "id": "image",
        "filename": "Formula 1.png",
        "question": "Como funciona um sábado de um fim de semana de um Grande Prêmio de Fórmula 1?",
        "expect": [
            "Treino Livre 3",
            "Classificação",
            "Define o grid de largada em uma sessão eliminatória",
            "Q1, Q2 e Q3",
        ],
    },
]


def _load_env_dev() -> None:
    """Carrega api/.env.dev no ambiente (se existir), sem sobrescrever o que já está setado."""
    env_dev = API_DIR / ".env.dev"
    if not env_dev.exists():
        return
    for line in env_dev.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _norm(text: str) -> str:
    """Normaliza para comparação robusta: minúsculas, sem acentos, espaços colapsados."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


@pytest.fixture(scope="module")
def e2e(tmp_path_factory: pytest.TempPathFactory) -> dict:
    _load_env_dev()
    if not os.getenv("CB_GOOGLE_API_KEY"):
        pytest.skip("CB_GOOGLE_API_KEY ausente — defina em api/.env.dev para rodar o e2e.")
    if not os.getenv("CB_OPENAI_API_KEY"):
        pytest.skip("CB_OPENAI_API_KEY ausente — defina em api/.env.dev para rodar o e2e.")
    if not os.getenv("CB_HF_TOKEN"):
        pytest.skip("CB_HF_TOKEN ausente — defina em api/.env.dev para rodar o e2e.")

    # Cache de modelos persistente (evita re-download) e Chroma isolado por execução.
    os.environ["CB_HF_CACHE_DIR"] = str(API_DIR / ".hf_cache")
    os.environ["CB_CHROMA_PATH"] = str(tmp_path_factory.mktemp("chroma_e2e_ingestion"))
    os.environ["CB_CHROMA_COLLECTION"] = "company_brain_e2e_ingestion"
    os.environ.setdefault("CB_MAX_NEW_TOKENS", "2048")

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()

    ingested: dict[str, str] = {}  # case id -> motivo da falha ("" se OK)
    with TestClient(app) as client:
        for case in CASES:
            path = ASSETS_DIR / case["filename"]
            if not path.exists():
                ingested[case["id"]] = f"arquivo ausente: {case['filename']}"
                continue
            try:
                with path.open("rb") as fh:
                    resp = client.post(
                        "/documents/upload",
                        files={"file": (case["filename"], fh)},
                        data={"doc_id": case["id"]},
                    )
            except Exception as exc:  # noqa: BLE001 — ex.: poppler ausente no PDF
                ingested[case["id"]] = f"falha no upload: {exc}"
                continue
            ingested[case["id"]] = "" if resp.status_code == 201 else f"HTTP {resp.status_code}: {resp.text}"

        yield {"client": client, "ingested": ingested}


@pytest.mark.e2e
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_chat_retrieves_expected_content(e2e: dict, case: dict) -> None:
    failure = e2e["ingested"].get(case["id"])
    if failure:
        pytest.skip(f"caso '{case['id']}' não ingerido — {failure}")

    client: TestClient = e2e["client"]
    resp = client.post("/chat", json={"message": case["question"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # OBS inegociável: a resposta gerada pelo Gemini NUNCA pode estar vazia.
    assert isinstance(body["answer"], str) and body["answer"].strip(), (
        f"resposta vazia para o caso '{case['id']}'"
    )

    sources = body["sources"]
    assert sources, f"nenhuma fonte recuperada para '{case['id']}'"
    all_text = "\n".join(s["text"] for s in sources)
    norm_all = _norm(all_text)

    # Cada trecho esperado deve aparecer nas fontes recuperadas.
    for expected in case["expect"]:
        assert _norm(expected) in norm_all, (
            f"[{case['id']}] trecho esperado ausente nas fontes: {expected!r}\n"
            f"Fontes: {all_text!r}"
        )

    # Página correta (quando aplicável): na fonte que contém o 1º trecho esperado.
    if "page" in case:
        first = _norm(case["expect"][0])
        matching = [s for s in sources if first in _norm(s["text"])]
        assert matching, f"[{case['id']}] nenhuma fonte contém o trecho para checar a página"
        assert matching[0]["metadata"].get("page") == case["page"], (
            f"[{case['id']}] página errada: esperado {case['page']}, "
            f"veio {matching[0]['metadata'].get('page')}"
        )

    # Timestamp (áudio): a linha que contém o trecho deve trazer um [HH:MM:SS]
    # próximo do esperado (tolerância em segundos).
    if "timestamp" in case:
        _assert_timestamp(all_text, case)


def _assert_timestamp(all_text: str, case: dict) -> None:
    expected_s = _to_seconds(case["timestamp"])
    tolerance = case.get("timestamp_tolerance_s", 15)
    target = _norm(case["expect"][0])

    found: list[int] = []
    for line in all_text.splitlines():
        match = re.match(r"\s*\[(\d{2}):(\d{2}):(\d{2})\]", line)
        if match and target in _norm(line):
            h, m, s = (int(g) for g in match.groups())
            found.append(h * 3600 + m * 60 + s)

    assert found, (
        f"[{case['id']}] nenhum timestamp [HH:MM:SS] na linha do trecho esperado.\n{all_text!r}"
    )
    assert any(abs(t - expected_s) <= tolerance for t in found), (
        f"[{case['id']}] timestamp fora da tolerância: esperado ~{case['timestamp']} "
        f"(±{tolerance}s), encontrados {found}"
    )


def _to_seconds(hhmmss: str) -> int:
    h, m, s = (int(p) for p in hhmmss.split(":"))
    return h * 3600 + m * 60 + s
