# Company Brain — API

Backend FastAPI do Company Brain: RAG multimodal sobre o conhecimento da empresa.

## Arquitetura

```
HTTP (FastAPI)
  ├── POST /documents          upsert de documento textual (JSON)
  ├── POST /documents/upload   upsert multimodal (texto, PDF, imagem, áudio)
  ├── DELETE /documents/{id}   remove um documento
  └── POST /chat               pergunta + histórico -> resposta com fontes

Orquestração (LlamaIndex)
  ├── GemmaEngine        google/gemma-4-E2B-it (carregado UMA vez; texto+imagem+áudio)
  ├── GemmaLLM           adaptador CustomLLM do LlamaIndex (geração de texto)
  ├── HuggingFaceEmbedding   microsoft/harrier-oss-v1-0.6b (instrução só na query)
  └── ChromaVectorStore  ChromaDB com PersistentClient (persistência local)
```

A `GemmaEngine` é a única dona do modelo Gemma 4 e é compartilhada entre o LLM de
chat e o processador multimodal de ingestão (STT de áudio e OCR/descrição de
imagens), evitando carregar o modelo duas vezes.

## Pré-requisitos

- Python 3.11+
- Token do Hugging Face com a licença do `google/gemma-4-E2B-it` aceita
  (modelo *gated*). Defina em `CB_HF_TOKEN`.
- GPU recomendada para inferência; CPU funciona, porém lento.

## Setup

```bash
cd api
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell: .venv\Scripts\Activate.ps1)
# source .venv/bin/activate       # Linux/macOS

pip install -e ".[dev]"
cp .env.example .env              # e edite CB_HF_TOKEN
```

## Rodar

```bash
python -m app.main
# ou
uvicorn app.main:app --reload
```

Docs interativas em `http://localhost:8000/docs`.

### Exemplos

```bash
# Adicionar/atualizar documento textual
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"manual-rh","text":"Funcionários têm 30 dias de férias.","metadata":{"area":"rh"}}'

# Upload multimodal (imagem/áudio/pdf são parseados pelo Gemma 4)
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@reuniao.wav" -F "doc_id=ata-2026-05"

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Quantos dias de férias eu tenho?"}'
```

## Testes

Os testes usam um `FakeRAGService` e **não** baixam modelos:

```bash
pytest
```

## Configuração

Todas as variáveis usam o prefixo `CB_` — veja [`.env.example`](.env.example).
Principais: `CB_HF_TOKEN`, `CB_LLM_MODEL`, `CB_EMBED_MODEL`, `CB_DEVICE`,
`CB_CHROMA_PATH`, `CB_SIMILARITY_TOP_K`, `CB_ENABLE_THINKING`.
