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
  ├── GemmaEngine        google/gemma-4-E2B-it (carregado UMA vez; texto+imagem)
  ├── WhisperEngine      openai/whisper-large-v3-turbo (STT multilíngue, c/ timestamps)
  ├── GemmaLLM           adaptador CustomLLM do LlamaIndex (geração de texto)
  ├── HuggingFaceEmbedding   microsoft/harrier-oss-v1-0.6b (instrução só na query)
  └── ChromaVectorStore  ChromaDB com PersistentClient (persistência local)
```

A `GemmaEngine` é a única dona do modelo Gemma 4 e é compartilhada entre o LLM de
chat e a camada de ingestão (OCR/descrição de imagens e páginas de PDF),
evitando carregar o modelo duas vezes. A transcrição de áudio fica a cargo da
`WhisperEngine` (Whisper multilíngue via Hugging Face), preservando timestamps.

### Ingestão por modalidade

Cada tipo de arquivo é roteado para a ferramenta adequada em `app/core/ingestion.py`:

| Tipo                | Ferramenta                                              | Saída                          |
| ------------------- | ------------------------------------------------------- | ------------------------------ |
| `.txt/.csv/.md/...` | leitura direta de texto                                 | 1 documento (`page=1`)         |
| `.pdf`              | **pdf2image** rasteriza + **Gemma 4** (OCR/descrição)   | 1 documento **por página**     |
| imagem (`.png/...`) | **Gemma 4** (visão: OCR + descrição)                    | 1 documento (`page=1`)         |
| áudio (`.mp3/...`)  | **Whisper** (`large-v3-turbo`, multilíngue) c/ timestamps | 1 documento (`[HH:MM:SS] ...`) |
| `.xlsx/.xlsm`       | **pandas + openpyxl** (texto) + **Gemma 4** (imagens)   | 1 documento **por aba**        |
| `.docx`             | **MarkItDown + python-docx** (imagens) + **Gemma 4**    | 1 documento (`page=1`)         |

## Pré-requisitos

- Python 3.11+
- Token do Hugging Face com a licença do `google/gemma-4-E2B-it` aceita
  (modelo *gated*). Defina em `CB_HF_TOKEN`.
- GPU recomendada para inferência; CPU funciona, porém lento.
- **poppler** (necessário pelo `pdf2image` para rasterizar PDFs):
  - Linux: `apt-get install poppler-utils`
  - macOS: `brew install poppler`
  - Windows: baixe em
    [poppler-windows](https://github.com/oschwartz10612/poppler-windows),
    extraia e aponte `CB_POPPLER_PATH` para a pasta `.../Library/bin` (ou
    adicione-a ao `PATH`).

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

A forma recomendada pela documentação atual do FastAPI é o CLI `fastapi`, que usa
o Uvicorn por baixo. O `entrypoint` (`app.main:app`) já está configurado no
`pyproject.toml`, então não é preciso passar o caminho:

```bash
fastapi dev      # desenvolvimento: auto-reload, escuta em 127.0.0.1
fastapi run      # produção: sem reload, escuta em 0.0.0.0
```

Alternativas equivalentes (chamando o Uvicorn diretamente):

```bash
python -m app.main                 # usa as Settings (CB_HOST/CB_PORT)
uvicorn app.main:app               # produção
uvicorn app.main:app --reload      # desenvolvimento
```

Docs interativas em `http://localhost:8000/docs`.

> **Windows.** O banner do CLI `fastapi` usa o emoji 🚀; em consoles legados
> (codec `cp1252`) isso causa `UnicodeEncodeError` antes do app subir. Rode com
> UTF-8 forçado: `set PYTHONUTF8=1` (cmd) ou `$env:PYTHONUTF8=1` (PowerShell)
> antes do `fastapi run`. Alternativa: usar `uvicorn app.main:app`, que não tem
> esse banner.

> **Produção / replicação.** Por padrão roda **1 processo**. Como o Gemma 4 E2B
> ocupa bastante VRAM/RAM e é carregado uma única vez no `lifespan`, prefira
> escalar com **1 worker por GPU** em múltiplas réplicas/containers (atrás de um
> proxy de terminação TLS) em vez de vários workers no mesmo processo — cada
> worker carregaria sua própria cópia do modelo.

### Exemplos

```bash
# Adicionar/atualizar documento textual
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"manual-rh","text":"Funcionários têm 30 dias de férias.","metadata":{"area":"rh"}}'

# Upload multimodal (áudio via Whisper; pdf/imagem/planilha/docx via Gemma 4)
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@reuniao.mp3" -F "doc_id=ata-2026-05"

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Quantos dias de férias eu tenho?"}'
```

## Testes

Os testes unitários usam um `FakeRAGService`/engines *fake* e **não** baixam modelos:

```bash
pytest
```

Os testes **end-to-end** (modelos reais: Gemma 4 + Whisper + harrier) ficam fora
da suíte padrão. Eles exigem `CB_HF_TOKEN` (em `.env.dev` ou no ambiente) e os
arquivos reais em [`tests/e2e_assets/`](tests/e2e_assets/README.md) (o caso de
PDF também precisa do poppler). Rode com:

```bash
pytest -m e2e -s
```

## Configuração

Todas as variáveis usam o prefixo `CB_` — veja [`.env.example`](.env.example).
Principais: `CB_HF_TOKEN`, `CB_LLM_MODEL`, `CB_EMBED_MODEL`, `CB_STT_MODEL`,
`CB_STT_LANGUAGE`, `CB_DEVICE`, `CB_CHROMA_PATH`, `CB_SIMILARITY_TOP_K`,
`CB_ENABLE_THINKING`, `CB_PDF_DPI`, `CB_POPPLER_PATH`.
