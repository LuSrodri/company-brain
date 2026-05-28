# Company Brain

Monorepo do **Company Brain** — uma base de conhecimento corporativa com RAG
multimodal, capaz de ingerir documentos, imagens e áudio e responder perguntas
sobre o conhecimento da empresa.

## Estrutura do monorepo

```
company-brain/
└── api/        # Backend FastAPI (Python) — RAG, ingestão e chat
```

Novos serviços (ex.: `web/`, `workers/`) serão adicionados como pastas irmãs de `api/`.

## Stack do /api

| Camada               | Tecnologia                                              |
| -------------------- | ------------------------------------------------------- |
| API HTTP             | FastAPI + Uvicorn                                       |
| Orquestração RAG     | LlamaIndex                                              |
| LLM / Multimodal     | `google/gemma-4-E2B-it` (texto, imagem, áudio) via HF   |
| Embeddings           | `microsoft/harrier-oss-v1-0.6b` via HF                  |
| Vector DB            | ChromaDB (persistência local em disco)                  |
| Testes               | pytest                                                  |

Veja [`api/README.md`](api/README.md) para instruções de execução.
