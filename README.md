# Company Brain

[![CI](https://github.com/LuSrodri/company-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/LuSrodri/company-brain/actions/workflows/ci.yml)

Base de conhecimento corporativa com **RAG multimodal**: ingere documentos,
planilhas, imagens e áudio e responde perguntas **citando a origem** (arquivo,
página, timestamp). Monorepo com backend FastAPI (`api/`) e um dashboard
React (`dashboard/`).

> Este README foca nos **problemas de engenharia** que o projeto resolve —
> alucinação, custo de inferência, latência e edge cases de ingestão. A lista de
> tecnologias está resumida ao final.

## Os desafios de engenharia (e como foram resolvidos)

### 1. Guardrails contra alucinação

Num assistente sobre dados internos, uma resposta inventada é pior do que
nenhuma resposta. Três camadas atacam isso:

- **Ancoragem obrigatória no contexto + citação.** O system prompt
  ([`api/app/core/rag.py`](api/app/core/rag.py)) restringe o modelo às passagens
  recuperadas, obriga citar a fonte (arquivo/página/timestamp) e manda **admitir
  quando não encontrou** em vez de completar. Áudio carrega `[HH:MM:SS]` no
  próprio texto, então a citação de timestamp é verificável.
- **Fontes retornadas na API.** Todo `/chat` devolve os `sources` (trecho, score
  e metadados) usados na resposta — o dashboard os exibe para **verificação
  humana**, transformando "confie no modelo" em "confira a fonte".
- **Medição, não intuição.** A [suíte de avaliação](api/evals/README.md) mede
  taxa de recusa fora de escopo e resistência a **prompt injection** embutida em
  documentos (ver §5).

### 2. Custo de inferência em escala

- **Skip por hash de conteúdo.** Cada documento tem um `content_hash` (SHA-256).
  Reenviar o mesmo `doc_id` com conteúdo idêntico **pula** OCR/STT/embeddings
  inteiros (`status: "unchanged"`) — o passo caro só roda quando o conteúdo muda.
  Em bases que recebem re-uploads frequentes, isso é o maior corte de custo.
- **Modelo dimensionado ao trabalho.** O chat/ingestão usa `gemini-3.1-flash-lite`
  (barato, otimizado para latência) com *thinking* mínimo; embeddings são
  encurtáveis via `CB_EMBED_DIMENSIONS` (Matryoshka) — troca marginal de acurácia
  por vetores menores e mais baratos.
- **Roadmap honesto.** O gargalo de custo/tempo na ingestão de PDF é o VLM
  **por página**. O plano pós-MVP é um caminho híbrido "texto-primeiro"
  (`pypdfium2`), chamando o VLM só nas páginas que realmente precisam de OCR.

### 3. Latência

O `/chat` é instrumentado: `RAGService.chat` mede a latência server-side
(recuperação + geração) e a expõe como `latency_ms` na resposta e no log. A
avaliação agrega **p50/p95** sobre várias chamadas, além do tempo de ingestão por
modalidade — números reprodutíveis em vez de impressão.

### 4. Edge cases de ingestão

Casos que quebram um RAG ingênuo, já tratados em
[`api/app/core/rag.py`](api/app/core/rag.py) e [`ingestion.py`](api/app/core/ingestion.py):

- **Chunks órfãos no update.** Reindexar uma versão com **menos** páginas deixaria
  chunks antigos órfãos. O delete é por metadado (`document == doc_id`), removendo
  todas as páginas/abas — não só o `doc_id` exato (coberto por teste).
- **Delete idempotente**, **hash em streaming** (arquivos grandes sem carregar
  tudo em memória) e **roteamento por modalidade** (PDF→página, planilha→aba,
  áudio→timestamp).
- **Encoding no Windows** (`cp1252` vs. emoji do CLI) documentado com contorno.

### 5. Métricas de qualidade (assertividade, segurança, latência)

A pasta [`api/evals/`](api/evals/README.md) mede o pipeline sobre **modelos e
arquivos reais** — o que os testes unitários (com fakes) não fazem:

| Dimensão      | Métrica                        | Como é medida                                              |
| ------------- | ------------------------------ | ---------------------------------------------------------- |
| Assertividade | Retrieval hit rate             | O trecho esperado veio nas fontes recuperadas (top-k)      |
| Assertividade | Correção (LLM-as-judge)        | Um 2º Gemini julga a resposta contra a referência          |
| Assertividade | Citação de fonte               | A resposta cita arquivo/página/timestamp                   |
| Segurança     | Recusa fora de escopo          | Perguntas sem resposta na base → o modelo recusa           |
| Segurança     | Resistência a prompt injection | Instrução maliciosa embutida em documento → é ignorada     |
| Latência      | Chat p50 / p95                 | `latency_ms` server-side por chamada                       |

```bash
cd api && python -m evals.run     # gera api/evals/results/latest.md (+ .json)
```

**Resultados da última execução** ([relatório completo](api/evals/results/latest.md)):

| Métrica                        | Resultado             |
| ------------------------------ | --------------------- |
| Retrieval hit rate             | 6/6 (100%)            |
| Correção (LLM-as-judge)        | 6/6 (100%)            |
| Citação de fonte               | 5/6 (83%)             |
| Recusa fora de escopo          | 4/4 (100%)            |
| Resistência a prompt injection | 1/1 (100%)            |
| Latência do chat (p50 / p95)   | 1332 ms / 2128 ms     |

O tempo de ingestão expõe o custo real do multimodal e sustenta o roadmap do §2:
**PDF ~210 s** e **áudio ~79 s** (VLM por página / STT), contra ~0,6 s de uma
planilha — é onde o caminho híbrido "texto-primeiro" corta mais.

## Qualidade e CI

- **CI no GitHub Actions** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):
  a cada push/PR roda `ruff` + `pytest` no `api/` (unitários, sem API keys — usam
  fakes) e `typecheck` + `build` no `dashboard/`. O badge acima reflete o estado
  do `main`. O job de API instala **torch CPU-only** para não baixar ~2 GB de CUDA.
- **Testes unitários** (`pytest`): rotas, gestão de documentos, devices e o
  `RAGService` contra um ChromaDB real com embeddings/LLM *mock* — rápidos e sem
  rede.
- **Testes e2e** (`pytest -m e2e`): pipeline real ponta a ponta (Gemini + Whisper
  + embeddings OpenAI), fora do CI por dependerem de chaves e arquivos.
- **Evals** (`python -m evals.run`): as métricas de qualidade acima.

## Estrutura do monorepo

```
company-brain/
├── api/        # Backend FastAPI (Python) — RAG, ingestão, chat e evals
└── dashboard/  # Frontend (Vite + React + TS) — chat e gestão de documentos
```

## Stack (resumo)

| Camada           | Tecnologia                                                    |
| ---------------- | ------------------------------------------------------------- |
| API HTTP         | FastAPI + Uvicorn                                             |
| Orquestração RAG | LlamaIndex                                                    |
| LLM / Multimodal | `gemini-3.1-flash-lite` (texto + imagem) via Google AI Studio |
| STT (áudio)      | `openai/whisper-large-v3-turbo` via HF (local)                |
| Embeddings       | `text-embedding-3-large` via OpenAI (API)                     |
| Vector DB        | ChromaDB (persistência local em disco)                        |
| Frontend         | Vite 6 · React 19 · TypeScript · Tailwind CSS v4 · lucide     |
| Qualidade        | pytest · ruff · GitHub Actions · suíte de evals               |

Detalhes de execução: [`api/README.md`](api/README.md) e
[`dashboard/README.md`](dashboard/README.md).
