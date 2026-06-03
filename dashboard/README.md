# Company Brain — Dashboard

Frontend do Company Brain: um chat RAG multimodal sobre a base de conhecimento da
empresa. Sidebar à esquerda para ingestão de documentos (drag & drop com os tipos
suportados) e o chat como elemento principal, com fontes citadas em cada resposta.

**Stack:** Vite 6 · React 19 · TypeScript · Tailwind CSS v4 · lucide-react ·
react-markdown. Tema claro "papel quente" (Fraunces + Plus Jakarta Sans).

## Pré-requisitos

A API (`../api`) precisa estar rodando em `http://localhost:8000`:

```bash
cd ../api
fastapi run app/main.py   # ou, com reload: fastapi dev app/main.py
```

> O dashboard consome a API via proxy do Vite (`/api/* → :8000`), então **não é
> necessário configurar CORS** no backend.

## Rodando o dashboard

```bash
npm install
npm run dev        # http://localhost:5173
```

Outros comandos:

```bash
npm run build      # typecheck (tsc) + build de produção em dist/
npm run preview    # serve o build de produção
npm run lint       # apenas typecheck
```

### Apontar para outra API

O alvo do proxy pode ser sobrescrito por variável de ambiente:

```bash
VITE_API_TARGET=http://192.168.0.10:8000 npm run dev
```

## O que está conectado à API

| Recurso na UI                | Endpoint                          |
| ---------------------------- | --------------------------------- |
| Enviar mensagem no chat      | `POST /chat`                      |
| Lista de documentos          | `GET /documents`                  |
| Upload (drag & drop / botão) | `POST /documents/upload`          |
| Remover documento            | `DELETE /documents/{doc_id}`      |
| Selo de status (online/off)  | `GET /health` (polling)           |

## Tipos de arquivo suportados

Espelham `SUPPORTED_EXTS` do backend (`api/app/core/ingestion.py`):

- **Texto** — `.txt .md .markdown .rst .csv .json`
- **PDF** — `.pdf`
- **Imagem** — `.png .jpg .jpeg .webp .gif .bmp`
- **Áudio** — `.wav .mp3 .flac .ogg .m4a`
- **Planilha** — `.xlsx .xlsm`
- **Documento** — `.docx`

## Estrutura

```
src/
  lib/        api.ts (client tipado), fileTypes.ts (modalidades/ícones/cores)
  hooks/      useChat, useDocuments, useHealth, useToast
  components/ Sidebar, Chat, Composer, MessageList, MessageBubble,
              Sources, Dropzone, DocumentList, EmptyState, Toaster, …
  index.css   design tokens (Tailwind v4 @theme) + tipografia do markdown
```
