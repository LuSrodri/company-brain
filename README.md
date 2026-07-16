# Company Brain

[![CI](https://github.com/LuSrodri/company-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/LuSrodri/company-brain/actions/workflows/ci.yml)

Monorepo do **Company Brain** — uma base de conhecimento corporativa com RAG
multimodal, capaz de ingerir documentos, imagens e áudio e responder perguntas
sobre o conhecimento da empresa.

## Estrutura do monorepo

```
company-brain/
├── api/        # Backend FastAPI (Python) — RAG, ingestão e chat
└── dashboard/  # Frontend (Vite + React + TS) — chat e gestão de documentos
```

## Stack do /api

| Camada               | Tecnologia                                              |
| -------------------- | ------------------------------------------------------- |
| API HTTP             | FastAPI + Uvicorn                                       |
| Orquestração RAG     | LlamaIndex                                              |
| LLM / Multimodal     | `gemini-3.1-flash-lite` (texto + imagem) via Google AI Studio |
| STT (áudio)          | `openai/whisper-large-v3-turbo` via HF (local)          |
| Embeddings           | `text-embedding-3-large` via OpenAI (API)               |
| Vector DB            | ChromaDB (persistência local em disco)                  |
| Testes               | pytest                                                  |

O chat responde **citando a origem** (arquivo, página, timestamp). Veja
[`api/README.md`](api/README.md) para execução do backend.

## Stack do /dashboard

Vite 6 · React 19 · TypeScript · Tailwind CSS v4 · lucide-react. Consome o `/api`
via proxy do Vite (sem CORS). Veja [`dashboard/README.md`](dashboard/README.md).
