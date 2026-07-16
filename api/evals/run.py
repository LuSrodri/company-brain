"""Executor da suíte de avaliação.

Sobe a API real (Gemini via Google AI Studio + Whisper local + embeddings
OpenAI), ingere os arquivos de ``tests/e2e_assets/``, roda os três grupos de
casos e escreve um relatório em ``evals/results/`` (Markdown + JSON).

    python -m evals.run              # a partir de api/
    python -m evals.run --keep-collection   # não apaga a coleção temporária

Requisitos (iguais aos do e2e): ``CB_GOOGLE_API_KEY``, ``CB_OPENAI_API_KEY`` e
``CB_HF_TOKEN`` no ambiente ou em ``api/.env.dev``; arquivos em
``tests/e2e_assets/``; poppler instalado (para o caso de PDF).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals import dataset
from evals.metrics import (
    LLMJudge,
    citation_present,
    is_refusal,
    normalize,
    percentiles,
    retrieval_hit,
)

API_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = API_DIR / "tests" / "e2e_assets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_env_dev() -> None:
    """Carrega api/.env.dev no ambiente (sem sobrescrever o que já está setado)."""
    env_dev = API_DIR / ".env.dev"
    if not env_dev.exists():
        return
    for line in env_dev.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _require_keys() -> None:
    missing = [
        name
        for name in ("CB_GOOGLE_API_KEY", "CB_OPENAI_API_KEY", "CB_HF_TOKEN")
        if not os.getenv(name)
    ]
    if missing:
        print(
            f"[evals] variáveis ausentes: {', '.join(missing)}.\n"
            f"Defina-as no ambiente ou em api/.env.dev para rodar a avaliação.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _build_judge() -> LLMJudge:
    """Constrói um LLM juiz independente (mesmo modelo, sem o system prompt de citação)."""
    from google.genai import types
    from llama_index.llms.google_genai import GoogleGenAI

    from app.config import get_settings

    s = get_settings()
    llm = GoogleGenAI(
        model=s.llm_model,
        api_key=s.google_api_key,
        context_window=1_000_000,
        max_tokens=1024,
        generation_config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
        ),
    )
    return LLMJudge(llm)


def _sources_text(body: dict[str, Any]) -> str:
    return "\n".join(s.get("text", "") for s in body.get("sources", []))


# Paceamento entre chamadas ao LLM: o tier do Gemini limita requisições por
# minuto; cada caso dispara geração + juiz. Um respiro curto entre chamadas +
# backoff no 429 mantém a avaliação estável sem exigir cota alta.
_PACE_SECONDS = float(os.getenv("CB_EVAL_PACE_SECONDS", "3"))
_BACKOFF_SECONDS = float(os.getenv("CB_EVAL_BACKOFF_SECONDS", "30"))
_MAX_RETRIES = int(os.getenv("CB_EVAL_MAX_RETRIES", "4"))


def _post_chat(client: Any, message: str) -> dict[str, Any]:
    """POST /chat com backoff em 429/5xx (rate limit do LLM). Retorna o corpo JSON."""
    for attempt in range(_MAX_RETRIES + 1):
        resp = client.post("/chat", json={"message": message})
        if resp.status_code == 200:
            time.sleep(_PACE_SECONDS)
            return resp.json()
        if resp.status_code in (429, 500, 503) and attempt < _MAX_RETRIES:
            wait = _BACKOFF_SECONDS * (attempt + 1)
            print(
                f"[evals] /chat HTTP {resp.status_code}; backoff {wait:.0f}s "
                f"(tentativa {attempt + 1}/{_MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue
        raise RuntimeError(f"/chat falhou (HTTP {resp.status_code}): {resp.text[:200]}")
    raise RuntimeError("/chat esgotou as tentativas de retry")


def _judge_with_retry(judge: LLMJudge, *, question: str, reference: str, answer: str) -> Any:
    """Chama o LLM-as-judge com backoff em erros transitórios (rate limit)."""
    from evals.metrics import JudgeVerdict

    for attempt in range(_MAX_RETRIES + 1):
        try:
            verdict = judge.judge(question=question, reference=reference, answer=answer)
            time.sleep(_PACE_SECONDS)
            return verdict
        except Exception as exc:  # noqa: BLE001 — trata 429/5xx do provider
            if attempt >= _MAX_RETRIES:
                return JudgeVerdict(correct=False, reason=f"juiz falhou: {exc}")
            wait = _BACKOFF_SECONDS * (attempt + 1)
            print(
                f"[evals] juiz falhou ({exc}); backoff {wait:.0f}s "
                f"(tentativa {attempt + 1}/{_MAX_RETRIES})",
                file=sys.stderr,
            )
            time.sleep(wait)
    return JudgeVerdict(correct=False, reason="juiz esgotou as tentativas de retry")


def run(keep_collection: bool = False) -> dict[str, Any]:
    _load_env_dev()
    _require_keys()

    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import create_app

    # Coleção temporária isolada, para não sujar a base de dev.
    os.environ["CB_HF_CACHE_DIR"] = str(API_DIR / ".hf_cache")
    os.environ["CB_CHROMA_PATH"] = str(API_DIR / ".chroma_evals")
    os.environ["CB_CHROMA_COLLECTION"] = "company_brain_evals"
    os.environ.setdefault("CB_MAX_NEW_TOKENS", "2048")
    get_settings.cache_clear()
    settings = get_settings()

    app = create_app()
    latencies: list[float] = []
    ingestion_ms: dict[str, float] = {}
    grounded_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []
    injection_rows: list[dict[str, Any]] = []

    # raise_server_exceptions=False: uma falha de ingestão num caso (ex.: poppler
    # ausente no PDF) vira um HTTP 500 tratável, sem abortar a avaliação inteira.
    print("[evals] subindo a API e carregando modelos...", file=sys.stderr)
    with TestClient(app, raise_server_exceptions=False) as client:
        judge = _build_judge()

        # ---------------- Ingestão dos assets (grounded) ---------------- #
        available: dict[str, bool] = {}
        for case in dataset.GROUNDED:
            path = ASSETS_DIR / case.filename
            if not path.exists():
                print(f"[evals] asset ausente, pulando: {case.filename}", file=sys.stderr)
                available[case.id] = False
                continue
            started = time.perf_counter()
            with path.open("rb") as fh:
                resp = client.post(
                    "/documents/upload",
                    files={"file": (case.filename, fh)},
                    data={"doc_id": f"eval-{case.id}"},
                )
            ingestion_ms[case.modality] = (time.perf_counter() - started) * 1000.0
            available[case.id] = resp.status_code == 201
            if resp.status_code != 201:
                print(f"[evals] falha ao ingerir {case.filename}: {resp.text}", file=sys.stderr)

        # Documento com injeção (texto puro).
        for inj in dataset.INJECTION:
            client.post("/documents", json={"doc_id": inj.doc_id, "text": inj.text})

        # ---------------- Assertividade ---------------- #
        print("[evals] avaliando assertividade...", file=sys.stderr)
        for case in dataset.GROUNDED:
            if not available.get(case.id):
                grounded_rows.append({"id": case.id, "modality": case.modality, "skipped": True})
                continue
            body = _post_chat(client, case.question)
            answer = body.get("answer", "")
            lat = body.get("latency_ms")
            if lat is not None:
                latencies.append(lat)
            hit = retrieval_hit(case.expect, _sources_text(body))
            cited = citation_present(answer, case.citation_hint)
            verdict = _judge_with_retry(
                judge,
                question=case.question,
                reference=" / ".join(case.expect),
                answer=answer,
            )
            grounded_rows.append(
                {
                    "id": case.id,
                    "modality": case.modality,
                    "skipped": False,
                    "retrieval_hit": hit,
                    "citation": cited,
                    "correct": verdict.correct,
                    "reason": verdict.reason,
                    "latency_ms": lat,
                }
            )

        # ---------------- Segurança: recusa fora de escopo ---------------- #
        print("[evals] avaliando segurança (recusa fora de escopo)...", file=sys.stderr)
        for oos in dataset.OUT_OF_SCOPE:
            body = _post_chat(client, oos.question)
            lat = body.get("latency_ms")
            if lat is not None:
                latencies.append(lat)
            refused = is_refusal(body.get("answer", ""))
            oos_rows.append({"id": oos.id, "refused": refused, "latency_ms": lat})

        # ---------------- Segurança: prompt injection ---------------- #
        print("[evals] avaliando segurança (prompt injection)...", file=sys.stderr)
        for inj in dataset.INJECTION:
            body = _post_chat(client, inj.question)
            answer = body.get("answer", "")
            lat = body.get("latency_ms")
            if lat is not None:
                latencies.append(lat)
            token_leaked = normalize(inj.injected_token) in normalize(answer)
            grounded_ok = retrieval_hit(inj.expect, _sources_text(body)) if inj.expect else True
            injection_rows.append(
                {
                    "id": inj.id,
                    "resisted": (not token_leaked) and grounded_ok,
                    "token_leaked": token_leaked,
                    "answered_from_source": grounded_ok,
                    "latency_ms": lat,
                }
            )

        # Limpeza da coleção temporária (a menos que --keep-collection).
        if not keep_collection:
            for case in dataset.GROUNDED:
                client.delete(f"/documents/eval-{case.id}")
            for inj in dataset.INJECTION:
                client.delete(f"/documents/{inj.doc_id}")

    report = _aggregate(
        settings=settings,
        grounded_rows=grounded_rows,
        oos_rows=oos_rows,
        injection_rows=injection_rows,
        latencies=latencies,
        ingestion_ms=ingestion_ms,
    )
    _write_reports(report)
    return report


def _rate(rows: list[dict[str, Any]], key: str) -> tuple[int, int]:
    """(acertos, total) considerando apenas linhas não puladas com a chave presente."""
    considered = [r for r in rows if not r.get("skipped") and key in r]
    hits = sum(1 for r in considered if r[key])
    return hits, len(considered)


def _aggregate(
    *,
    settings: Any,
    grounded_rows: list[dict[str, Any]],
    oos_rows: list[dict[str, Any]],
    injection_rows: list[dict[str, Any]],
    latencies: list[float],
    ingestion_ms: dict[str, float],
) -> dict[str, Any]:
    hit_n, hit_d = _rate(grounded_rows, "retrieval_hit")
    cite_n, cite_d = _rate(grounded_rows, "citation")
    corr_n, corr_d = _rate(grounded_rows, "correct")
    refused_n = sum(1 for r in oos_rows if r["refused"])
    resisted_n = sum(1 for r in injection_rows if r["resisted"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "llm_model": settings.llm_model,
            "embed_model": settings.embed_model,
            "embed_dimensions": settings.embed_dimensions,
            "similarity_top_k": settings.similarity_top_k,
        },
        "assertividade": {
            "retrieval_hit_rate": {"hits": hit_n, "total": hit_d},
            "citation_rate": {"hits": cite_n, "total": cite_d},
            "answer_correct_rate": {"hits": corr_n, "total": corr_d},
            "cases": grounded_rows,
        },
        "seguranca": {
            "refusal_rate": {"hits": refused_n, "total": len(oos_rows)},
            "injection_resistance": {"hits": resisted_n, "total": len(injection_rows)},
            "out_of_scope_cases": oos_rows,
            "injection_cases": injection_rows,
        },
        "latencia_ms": percentiles(latencies),
        "ingestao_ms": ingestion_ms,
    }


def _pct(hits: int, total: int) -> str:
    return f"{hits}/{total} ({100.0 * hits / total:.0f}%)" if total else "n/a"


def _render_markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    a = report["assertividade"]
    s = report["seguranca"]
    lat = report["latencia_ms"]

    lines: list[str] = []
    lines.append("# Relatório de avaliação — Company Brain")
    lines.append("")
    lines.append(f"_Gerado em {report['generated_at']}_")
    lines.append("")
    dims = f" ({cfg['embed_dimensions']}d)" if cfg["embed_dimensions"] else ""
    lines.append(
        f"**Config:** LLM `{cfg['llm_model']}` · embeddings `{cfg['embed_model']}`{dims}"
        f" · top_k `{cfg['similarity_top_k']}`"
    )
    lines.append("")

    # Sumário
    lines.append("## Sumário")
    lines.append("")
    lines.append("| Dimensão | Métrica | Resultado |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Assertividade | Retrieval hit rate | "
        f"{_pct(**a['retrieval_hit_rate'])} |"
    )
    lines.append(
        f"| Assertividade | Correção (LLM-judge) | {_pct(**a['answer_correct_rate'])} |"
    )
    lines.append(f"| Assertividade | Citação de fonte | {_pct(**a['citation_rate'])} |")
    lines.append(f"| Segurança | Recusa fora de escopo | {_pct(**s['refusal_rate'])} |")
    lines.append(
        f"| Segurança | Resistência a injeção | {_pct(**s['injection_resistance'])} |"
    )
    lines.append(
        f"| Latência | Chat p50 / p95 | {lat['p50']:.0f} ms / {lat['p95']:.0f} ms |"
    )
    lines.append("")

    # Assertividade detalhada
    lines.append("## Assertividade (por modalidade)")
    lines.append("")
    lines.append("| Caso | Modalidade | Retrieval | Citou | Correto | Latência |")
    lines.append("| --- | --- | :---: | :---: | :---: | ---: |")
    for row in a["cases"]:
        if row.get("skipped"):
            lines.append(f"| {row['id']} | {row['modality']} | — | — | — | _asset ausente_ |")
            continue
        lat_ms = f"{row['latency_ms']:.0f} ms" if row.get("latency_ms") is not None else "—"
        lines.append(
            f"| {row['id']} | {row['modality']} | {_yn(row['retrieval_hit'])} | "
            f"{_yn(row['citation'])} | {_yn(row['correct'])} | {lat_ms} |"
        )
    lines.append("")

    # Segurança
    lines.append("## Segurança")
    lines.append("")
    lines.append("**Recusa fora de escopo** (perguntas sem resposta na base):")
    lines.append("")
    lines.append("| Pergunta (id) | Recusou? |")
    lines.append("| --- | :---: |")
    for row in s["out_of_scope_cases"]:
        lines.append(f"| {row['id']} | {_yn(row['refused'])} |")
    lines.append("")
    lines.append("**Prompt injection** (instrução maliciosa embutida em documento):")
    lines.append("")
    lines.append("| Caso | Resistiu? | Vazou token? | Respondeu da fonte? |")
    lines.append("| --- | :---: | :---: | :---: |")
    for row in s["injection_cases"]:
        lines.append(
            f"| {row['id']} | {_yn(row['resisted'])} | {_yn(row['token_leaked'])} | "
            f"{_yn(row['answered_from_source'])} |"
        )
    lines.append("")

    # Latência
    lines.append("## Latência e custo")
    lines.append("")
    lines.append(
        f"Chat (recuperação + geração), {lat['n']} chamadas: "
        f"**p50 {lat['p50']:.0f} ms**, **p95 {lat['p95']:.0f} ms**, "
        f"média {lat['mean']:.0f} ms, máx {lat['max']:.0f} ms."
    )
    lines.append("")
    if report["ingestao_ms"]:
        lines.append("Tempo de ingestão por modalidade (1 arquivo cada):")
        lines.append("")
        lines.append("| Modalidade | Ingestão |")
        lines.append("| --- | ---: |")
        for modality, ms in report["ingestao_ms"].items():
            lines.append(f"| {modality} | {ms:.0f} ms |")
        lines.append("")

    return "\n".join(lines)


def _yn(value: bool) -> str:
    return "✅" if value else "❌"


def _write_reports(report: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = _render_markdown(report)
    (RESULTS_DIR / "latest.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(md)
    print(f"\n[evals] relatório salvo em {RESULTS_DIR / 'latest.md'}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suíte de avaliação do Company Brain.")
    parser.add_argument(
        "--keep-collection",
        action="store_true",
        help="Não remove os documentos/coleção temporários da avaliação ao final.",
    )
    args = parser.parse_args()
    run(keep_collection=args.keep_collection)


if __name__ == "__main__":
    main()
