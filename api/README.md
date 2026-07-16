# Company Brain — API

[![CI](https://github.com/LuSrodri/company-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/LuSrodri/company-brain/actions/workflows/ci.yml)

Backend FastAPI do Company Brain: RAG multimodal sobre o conhecimento da empresa.

> Panorama dos **desafios de engenharia** (alucinação, custo, latência, edge
> cases) e das **métricas de qualidade** no [README do monorepo](../README.md).
> Aqui: arquitetura, execução e configuração.

## Arquitetura

```
HTTP (FastAPI)
  ├── GET  /documents          lista os documentos indexados (agrupados por doc_id)
  ├── POST /documents          upsert de documento textual (JSON)
  ├── POST /documents/upload   upsert multimodal (texto, PDF, imagem, áudio)
  ├── DELETE /documents/{id}   remove um documento inteiro (todas as páginas/abas)
  └── POST /chat               pergunta + histórico -> resposta com fontes

Orquestração (LlamaIndex)
  ├── GoogleGenAI        gemini-3.1-flash-lite via Google AI Studio (chat; texto+imagem)
  ├── ImageDescriber     usa o GoogleGenAI p/ OCR/descrição de imagens na ingestão
  ├── WhisperEngine      openai/whisper-large-v3-turbo (STT local, c/ timestamps)
  ├── OpenAIEmbedding    text-embedding-3-large (API; 3072 dims)
  └── ChromaVectorStore  ChromaDB com PersistentClient (persistência local)
```

O LLM é o `GoogleGenAI` da integração **oficial** `llama-index-llms-google-genai`
(wrap do SDK `google-genai`). O **mesmo** LLM serve o chat e a descrição
multimodal de imagens da ingestão (via `ImageDescriber`, que monta uma
`ChatMessage` com `ImageBlock` + `TextBlock`). A transcrição de áudio fica a cargo
da `WhisperEngine` (Whisper multilíngue via Hugging Face, local), com timestamps.
Os **embeddings** são servidos pela API da OpenAI (`text-embedding-3-large`, 3072
dims; encurtáveis via `CB_EMBED_DIMENSIONS`) — o mesmo modelo embeda documentos e
queries, sem instrução nem device.

### Device do modelo local (NVIDIA / AMD / CPU)

LLM/ingestão (Google AI Studio) e embeddings (OpenAI) rodam na nuvem. O único
modelo que ainda roda localmente — **Whisper (STT)** — escolhe o device via
`CB_DEVICE` (`app/core/devices.py`):

- `auto` (padrão): `cuda` → `mps` → `cpu`.
- **NVIDIA**: CUDA (`cuda`).
- **AMD**: ROCm, que no PyTorch é exposto como `cuda` (use `cuda` ou o alias
  `rocm`). Cobre GPUs suportadas pelo ROCm; GPUs AMD antigas **não** cobertas
  pelo ROCm (ex.: RX 580 / Polaris) caem em `cpu` — não há suporte a DirectML
  (ele prenderia o PyTorch em 2.3.1, incompatível com o stack atual).
- **CPU**: `cpu` (fallback sempre disponível).

### Gerenciamento de documentos

Cada documento lógico é identificado por um `doc_id` estável e gravado nos
metadados de todos os seus chunks (`document`), junto de `source`, `modality`,
`page`/`sheet` e `content_hash`. Isso sustenta três operações:

- **Listar** (`GET /documents`): agrega os chunks por `doc_id`, devolvendo
  fonte, modalidade, páginas e contagem de chunks.
- **Excluir** (`DELETE /documents/{id}`): remove **todas** as páginas/abas do
  documento de uma vez (filtro por metadado `document`), sem deixar chunks
  órfãos.
- **Update inteligente** (reenviar o mesmo `doc_id`): o `content_hash`
  (SHA-256 do conteúdo) é comparado antes de processar. Se o arquivo é idêntico,
  a ingestão é **pulada** (`status: "unchanged"`), evitando re-rodar OCR/STT/
  embeddings. Se mudou, a versão anterior é substituída por completo
  (`status: "upserted"`).

### Ingestão por modalidade

Cada tipo de arquivo é roteado para a ferramenta adequada em `app/core/ingestion.py`:

| Tipo                | Ferramenta                                              | Saída                          |
| ------------------- | ------------------------------------------------------- | ------------------------------ |
| `.txt/.csv/.md/...` | leitura direta de texto                                 | 1 documento (`page=1`)         |
| `.pdf`              | **pdf2image** rasteriza + **Gemini 3.1 Flash-Lite** (API, OCR/descrição) | 1 documento **por página**     |
| imagem (`.png/...`) | **Gemini 3.1 Flash-Lite** (API, visão: OCR + descrição)               | 1 documento (`page=1`)         |
| áudio (`.mp3/...`)  | **Whisper** (`large-v3-turbo`, multilíngue) c/ timestamps | 1 documento (`[HH:MM:SS] ...`) |
| `.xlsx/.xlsm`       | **pandas + openpyxl** (texto) + **Gemini 3.1 Flash-Lite** (API, imagens) | 1 documento **por aba**        |
| `.docx`             | **MarkItDown + python-docx** (imagens) + **Gemini 3.1 Flash-Lite** (API) | 1 documento (`page=1`)         |

## Pré-requisitos

- Python 3.11+
- **Chave do Google AI Studio** (`CB_GOOGLE_API_KEY`) para o LLM/ingestão
  multimodal (`gemini-3.1-flash-lite`). Gere em <https://aistudio.google.com/apikey>.
- **Chave da OpenAI** (`CB_OPENAI_API_KEY`) para os embeddings
  (`text-embedding-3-large`). Gere em <https://platform.openai.com/api-keys>.
- Token do Hugging Face (`CB_HF_TOKEN`) — opcional, usado só pelo modelo local
  (Whisper STT).
- GPU recomendada para o Whisper; CPU funciona, porém lento. Veja a seção
  *Device* sobre NVIDIA/AMD/CPU.
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
cp .env.example .env              # e edite CB_GOOGLE_API_KEY (e CB_HF_TOKEN se preciso)
```

> **O `.env` é obrigatório.** A app lê **apenas** `.env` (veja `env_file` em
> `app/config.py`); arquivos como `.env.dev`/`.env.local` **não** são carregados
> automaticamente — só por scripts/testes que os importam de propósito. Todos os
> `.env*` (menos `.env.example`) estão no `.gitignore`, então em cada checkout/
> máquina você precisa criar o seu `.env`. Sem a `CB_GOOGLE_API_KEY` no `.env`, o
> startup falha já no `lifespan` com:
>
> ```
> ValueError: No API key was provided. Please pass a valid API key.
> ```
>
> Se você mantém um `.env.dev`, basta `cp .env.dev .env` para reaproveitar as chaves.

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

> **Produção / replicação.** LLM (Google AI Studio) e embeddings (OpenAI) rodam
> na nuvem, então o processo local carrega apenas o Whisper. Ainda assim, ele
> ocupa RAM/VRAM e é carregado uma vez no `lifespan`; ao escalar, prefira **1
> worker por GPU** em réplicas/containers (atrás de um proxy TLS) a vários
> workers no mesmo processo, e proteja suas cotas de API contra concorrência alta.

### Exemplos

```bash
# Adicionar/atualizar documento textual
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"manual-rh","text":"Funcionários têm 30 dias de férias.","metadata":{"area":"rh"}}'

# Upload multimodal (áudio via Whisper; pdf/imagem/planilha/docx via Gemini 3.1 Flash-Lite)
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@reuniao.mp3" -F "doc_id=ata-2026-05"

# Listar documentos indexados
curl http://localhost:8000/documents

# Remover um documento inteiro (todas as páginas/abas)
curl -X DELETE http://localhost:8000/documents/ata-2026-05

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Quantos dias de férias eu tenho?"}'
```

## Guardrails e observabilidade

- **Anti-alucinação.** O system prompt (`app/core/rag.py`) prende o modelo ao
  contexto recuperado, obriga citar a fonte e manda dizer que **não encontrou**
  em vez de inventar. Toda resposta de `/chat` devolve os `sources` que a
  sustentam, para verificação humana.
- **Latência.** `RAGService.chat` mede a latência server-side (recuperação +
  geração) e a expõe como `latency_ms` no corpo da resposta e no log — base para
  os percentis p50/p95 da avaliação.

## Testes

Três camadas, do mais rápido ao mais caro:

**1. Unitários** — usam `FakeRAGService`/engines *fake* (ou um ChromaDB real com
LLM/embeddings *mock*) e **não** baixam modelos nem exigem chaves. É o que roda no
CI a cada push:

```bash
pytest
```

**2. End-to-end** (`-m e2e`) — pipeline real: `gemini-3.1-flash-lite` via API +
Whisper local + embeddings OpenAI. Exigem `CB_GOOGLE_API_KEY`,
`CB_OPENAI_API_KEY` e `CB_HF_TOKEN` (em `.env.dev` ou no ambiente) e os arquivos
reais em [`tests/e2e_assets/`](tests/e2e_assets/README.md) (o caso de PDF também
precisa do poppler):

```bash
pytest -m e2e -s
```

**3. Avaliação de qualidade** ([`evals/`](evals/README.md)) — mede
assertividade (retrieval hit rate, correção via LLM-as-judge, citação),
segurança (recusa fora de escopo, resistência a prompt injection) e latência
(p50/p95). Mesmos requisitos do e2e; gera `evals/results/latest.md` e `.json`:

```bash
python -m evals.run
```

## Configuração

Todas as variáveis usam o prefixo `CB_` — veja [`.env.example`](.env.example).
Principais: `CB_GOOGLE_API_KEY`, `CB_OPENAI_API_KEY`, `CB_LLM_MODEL`,
`CB_EMBED_MODEL`, `CB_EMBED_DIMENSIONS`, `CB_EMBED_BATCH_SIZE`, `CB_HF_TOKEN`,
`CB_STT_MODEL`, `CB_STT_LANGUAGE`, `CB_DEVICE`, `CB_CHROMA_PATH`,
`CB_SIMILARITY_TOP_K`, `CB_MAX_NEW_TOKENS`, `CB_PDF_DPI`, `CB_POPPLER_PATH`.
