"""Teste end-to-end REAL (modelos de verdade) do fluxo de RAG via /chat, com PDFs.

Diferente de injetar metadata de página na mão, aqui geramos **PDFs reais**
multi-página, enviamos pelo ``POST /documents/upload`` e deixamos o pipeline
(pypdf → 1 documento por página) atribuir o número da página. Depois, uma
pergunta por documento via ``POST /chat`` deve recuperar o **parágrafo** e a
**página** corretos — agora a página vem do parser, não do teste.

Pesado: baixa/carrega Gemma 4 E2B (gated) + harrier. Fora da suíte padrão:

    pytest -m e2e -s

Requisitos: ``CB_HF_TOKEN`` no ambiente OU em ``api/.env.dev``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

API_DIR = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# 4 documentos, cada um com várias páginas (1 parágrafo por página).
# `answer_page` é a página (1-based) onde está a resposta; `expect_substring`
# é ASCII para sobreviver ao round-trip de extração do pypdf.
# --------------------------------------------------------------------------- #
DOCUMENTS: list[dict] = [
    {
        "doc_id": "manual-rh",
        "title": "Manual de RH",
        "pages": [
            "Este manual descreve as politicas de recursos humanos da empresa, "
            "incluindo admissao, conduta e desligamento de colaboradores.",
            "Todo colaborador com contrato CLT tem direito a 30 dias corridos de "
            "ferias remuneradas a cada periodo de 12 meses de trabalho.",
            "O vale-refeicao e creditado no primeiro dia util de cada mes e o "
            "plano de saude cobre o colaborador e seus dependentes diretos.",
        ],
        "question": "Quantos dias de ferias por ano um colaborador CLT tem direito?",
        "answer_page": 2,
        "expect_substring": "30 dias",
    },
    {
        "doc_id": "seguranca",
        "title": "Politica de Seguranca da Informacao",
        "pages": [
            "Esta politica define as regras de protecao da informacao e se aplica "
            "a todos os sistemas e dados corporativos da empresa.",
            "As senhas de acesso devem ter no minimo 12 caracteres, combinando "
            "letras maiusculas, minusculas, numeros e simbolos.",
            "Todo acesso a sistemas criticos exige autenticacao em duas etapas.",
            "Incidentes de seguranca devem ser comunicados ao time de TI em ate "
            "uma hora apos a deteccao.",
        ],
        "question": "Qual o tamanho minimo exigido para as senhas de acesso?",
        "answer_page": 2,
        "expect_substring": "12 caracteres",
    },
    {
        "doc_id": "reembolso",
        "title": "Politica de Reembolso de Despesas",
        "pages": [
            "Despesas corporativas elegiveis incluem viagens a trabalho, "
            "hospedagem e alimentacao durante deslocamentos.",
            "O pedido de reembolso deve ser enviado em ate 30 dias apos a data da "
            "despesa, anexando a nota fiscal correspondente.",
        ],
        "question": "Qual o prazo para enviar um pedido de reembolso de despesa?",
        "answer_page": 2,
        "expect_substring": "30 dias",
    },
    {
        "doc_id": "remoto",
        "title": "Guia de Trabalho Remoto",
        "pages": [
            "O trabalho remoto e permitido para funcoes compativeis, mediante "
            "aprovacao do gestor direto.",
            "A empresa fornece notebook e auxilio para internet aos colaboradores "
            "em regime remoto.",
            "A ferramenta oficial para reunioes remotas e o Google Meet, e o "
            "Slack e o canal padrao para comunicacao assincrona.",
        ],
        "question": "Qual e a ferramenta oficial para reunioes remotas?",
        "answer_page": 3,
        "expect_substring": "Google Meet",
    },
]


def _load_env_dev() -> None:
    """Carrega api/.env.dev no ambiente (se existir e o token ainda não estiver setado)."""
    if os.getenv("CB_HF_TOKEN"):
        return
    env_dev = API_DIR / ".env.dev"
    if not env_dev.exists():
        return
    for line in env_dev.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _write_pdf(path: Path, title: str, pages: list[str]) -> None:
    """Gera um PDF real: uma página por item de ``pages``."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=14)
        pdf.multi_cell(0, 10, f"{title}\n\n{page_text}")
    pdf.output(str(path))


@pytest.fixture(scope="module")
def e2e_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    _load_env_dev()
    if not os.getenv("CB_HF_TOKEN"):
        pytest.skip("CB_HF_TOKEN ausente — defina em api/.env.dev para rodar o e2e.")

    # Reutiliza o cache de modelos persistente (evita re-download) e Chroma isolado.
    os.environ["CB_HF_CACHE_DIR"] = str(API_DIR / ".hf_cache")
    os.environ["CB_CHROMA_PATH"] = str(tmp_path_factory.mktemp("chroma_e2e_pdf"))
    os.environ["CB_CHROMA_COLLECTION"] = "company_brain_e2e_pdf"
    os.environ["CB_MAX_NEW_TOKENS"] = "48"

    # Gera os PDFs reais em um diretório temporário.
    pdf_dir = tmp_path_factory.mktemp("pdfs")
    pdf_paths: dict[str, Path] = {}
    for doc in DOCUMENTS:
        path = pdf_dir / f"{doc['doc_id']}.pdf"
        _write_pdf(path, doc["title"], doc["pages"])
        pdf_paths[doc["doc_id"]] = path

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        # Upload de cada PDF: o pipeline cria 1 documento por página.
        for doc in DOCUMENTS:
            with pdf_paths[doc["doc_id"]].open("rb") as fh:
                resp = client.post(
                    "/documents/upload",
                    files={"file": (f"{doc['doc_id']}.pdf", fh, "application/pdf")},
                    data={"doc_id": doc["doc_id"]},
                )
            assert resp.status_code == 201, resp.text
            # Confirma que houve uma página por documento (doc_ids tipo "manual-rh:p2").
            assert len(resp.json()["doc_ids"]) == len(doc["pages"]), resp.text
        yield client


@pytest.mark.e2e
@pytest.mark.parametrize("case", DOCUMENTS, ids=[d["doc_id"] for d in DOCUMENTS])
def test_chat_retrieves_correct_paragraph_and_page(e2e_client: TestClient, case: dict) -> None:
    resp = e2e_client.post("/chat", json={"message": case["question"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # O chat (Gemma 4) deve produzir alguma resposta...
    assert isinstance(body["answer"], str) and body["answer"].strip()

    # ...mas "parágrafo/página corretos" é verificado sobre as FONTES recuperadas,
    # cuja página foi extraída do PDF pelo pypdf (não injetada pelo teste).
    sources = body["sources"]
    assert sources, "nenhuma fonte recuperada"

    top = sources[0]
    meta = top["metadata"]
    assert meta["document"] == case["doc_id"], (
        f"documento errado: esperado {case['doc_id']}, veio {meta.get('document')}"
    )
    assert meta["page"] == case["answer_page"], (
        f"pagina errada para {case['doc_id']}: esperado {case['answer_page']}, "
        f"veio {meta.get('page')}"
    )
    assert case["expect_substring"].lower() in top["text"].lower(), (
        f"paragrafo errado: '{case['expect_substring']}' nao esta na fonte top-1: {top['text']!r}"
    )
