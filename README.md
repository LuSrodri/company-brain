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
| LLM / Multimodal     | `gemma-4-31b-it` (texto + imagem) via Google AI Studio  |
| STT (áudio)          | `openai/whisper-large-v3-turbo` via HF (local)          |
| Embeddings           | `microsoft/harrier-oss-v1-0.6b` via HF (local)          |
| Vector DB            | ChromaDB (persistência local em disco)                  |
| Testes               | pytest                                                  |

Veja [`api/README.md`](api/README.md) para instruções de execução.
