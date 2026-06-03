import { useState } from "react";
import { Loader2, Trash2 } from "lucide-react";
import type { DocumentInfo } from "../lib/api";
import { modalityMeta } from "../lib/fileTypes";

export function DocumentItem({
  doc,
  onDelete,
}: {
  doc: DocumentInfo;
  onDelete: (doc: DocumentInfo) => Promise<void>;
}) {
  const meta = modalityMeta(doc.modality);
  const Icon = meta.icon;
  const [removing, setRemoving] = useState(false);

  // O `doc_id` carrega o nome do arquivo original (no upload, é o file.filename).
  // O campo `source` da API guarda o nome temporário com UUID, então não usamos.
  const title = doc.doc_id || doc.source || "Documento";
  const pageLabel =
    doc.pages.length > 1 ? `${doc.pages.length} págs` : doc.pages.length === 1 ? "1 pág" : null;

  async function handleDelete() {
    setRemoving(true);
    try {
      await onDelete(doc);
    } finally {
      setRemoving(false);
    }
  }

  return (
    <li className="group flex items-center gap-3 rounded-xl border border-transparent px-2 py-2 transition-colors hover:border-line hover:bg-paper">
      <span className={`grid size-9 shrink-0 place-items-center rounded-xl ${meta.bg} ${meta.fg}`}>
        <Icon className="size-4.5" strokeWidth={2.2} />
      </span>

      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold text-ink" title={title}>
          {title}
        </p>
        <p className="flex items-center gap-1.5 text-xs text-muted">
          <span className={`font-medium ${meta.fg}`}>{meta.label}</span>
          <span className="text-faint">·</span>
          <span>{doc.chunks} chunks</span>
          {pageLabel && (
            <>
              <span className="text-faint">·</span>
              <span>{pageLabel}</span>
            </>
          )}
        </p>
      </div>

      <button
        onClick={handleDelete}
        disabled={removing}
        className="shrink-0 rounded-lg p-1.5 text-faint opacity-0 transition hover:bg-accent-soft hover:text-accent-deep focus-visible:opacity-100 group-hover:opacity-100 disabled:opacity-100"
        aria-label={`Remover ${title}`}
        title="Remover documento"
      >
        {removing ? (
          <Loader2 className="size-4 animate-spin" />
        ) : (
          <Trash2 className="size-4" />
        )}
      </button>
    </li>
  );
}
