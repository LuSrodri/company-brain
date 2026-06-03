/**
 * Client tipado da Company Brain API.
 *
 * Em dev, todas as chamadas usam o prefixo `/api`, que o Vite faz proxy para
 * http://localhost:8000 (ver vite.config.ts) — sem necessidade de CORS no backend.
 */

const BASE = "/api";

export type ChatRole = "user" | "assistant" | "system";

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export interface Source {
  text: string;
  score: number | null;
  metadata: Record<string, unknown>;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
}

export type Modality =
  | "text"
  | "pdf"
  | "image"
  | "audio"
  | "spreadsheet"
  | "document";

export interface DocumentInfo {
  doc_id: string;
  source: string | null;
  modality: string | null;
  pages: number[];
  chunks: number;
  content_hash: string | null;
}

export interface DocumentList {
  documents: DocumentInfo[];
  total_documents: number;
  total_chunks: number;
}

export interface DocumentResponse {
  doc_ids: string[];
  status: string;
  total_chunks: number;
}

export interface HealthResponse {
  status: string;
  version: string;
}

/** Erro de API com status HTTP e mensagem amigável extraída do corpo. */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(res: Response): Promise<never> {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
      detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    }
  } catch {
    /* corpo não-JSON: mantém o status */
  }
  throw new ApiError(detail, res.status);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(
      "Não foi possível conectar à API. Ela está rodando em :8000?",
      0,
    );
  }
  if (!res.ok) await parseError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: (signal?: AbortSignal) =>
    request<HealthResponse>("/health", { signal }),

  listDocuments: (signal?: AbortSignal) =>
    request<DocumentList>("/documents", { signal }),

  uploadDocument: (
    file: File,
    opts?: { docId?: string; metadata?: Record<string, unknown> },
  ) => {
    const form = new FormData();
    form.append("file", file);
    if (opts?.docId) form.append("doc_id", opts.docId);
    if (opts?.metadata) form.append("metadata", JSON.stringify(opts.metadata));
    return request<DocumentResponse>("/documents/upload", {
      method: "POST",
      body: form,
    });
  },

  deleteDocument: (docId: string) =>
    request<void>(`/documents/${encodeURIComponent(docId)}`, {
      method: "DELETE",
    }),

  chat: (
    message: string,
    history: ChatTurn[],
    opts?: { topK?: number; signal?: AbortSignal },
  ) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history,
        ...(opts?.topK ? { top_k: opts.topK } : {}),
      }),
      signal: opts?.signal,
    }),
};
