import { Library, Loader2, RefreshCw } from "lucide-react";
import type { DocumentInfo } from "../lib/api";
import { DocumentItem } from "./DocumentItem";

export function DocumentList({
  documents,
  loading,
  error,
  totalChunks,
  onDelete,
  onRefresh,
}: {
  documents: DocumentInfo[];
  loading: boolean;
  error: string | null;
  totalChunks: number;
  onDelete: (doc: DocumentInfo) => Promise<void>;
  onRefresh: () => void;
}) {
  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <header className="mb-2 flex items-center justify-between px-2">
        <div className="flex items-center gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-faint">
            Base de conhecimento
          </h2>
          {documents.length > 0 && (
            <span className="rounded-full bg-canvas px-1.5 py-0.5 text-[10px] font-semibold text-muted">
              {documents.length}
            </span>
          )}
        </div>
        <button
          onClick={onRefresh}
          className="rounded-lg p-1 text-faint transition hover:bg-paper hover:text-ink"
          aria-label="Atualizar lista"
          title="Atualizar"
        >
          <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        {loading && documents.length === 0 ? (
          <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted">
            <Loader2 className="size-4 animate-spin" />
            Carregando documentos…
          </div>
        ) : error ? (
          <p className="rounded-xl border border-accent-soft bg-accent-soft/50 px-3 py-2.5 text-xs text-accent-deep">
            {error}
          </p>
        ) : documents.length === 0 ? (
          <div className="flex flex-col items-center gap-2 px-4 py-8 text-center">
            <span className="grid size-11 place-items-center rounded-2xl bg-canvas text-faint">
              <Library className="size-5" />
            </span>
            <p className="text-sm font-medium text-muted">Nenhum documento ainda</p>
            <p className="text-xs text-faint">
              Envie um arquivo acima para alimentar o cérebro da empresa.
            </p>
          </div>
        ) : (
          <ul className="space-y-0.5">
            {documents.map((doc) => (
              <DocumentItem key={doc.doc_id} doc={doc} onDelete={onDelete} />
            ))}
          </ul>
        )}
      </div>

      {documents.length > 0 && (
        <p className="mt-2 border-t border-line px-2 pt-2.5 text-xs text-faint">
          <span className="font-semibold text-muted">{totalChunks.toLocaleString("pt-BR")}</span>{" "}
          chunks indexados no total
        </p>
      )}
    </section>
  );
}
