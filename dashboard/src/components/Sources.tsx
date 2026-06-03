import { useState } from "react";
import { ChevronDown, Quote } from "lucide-react";
import type { Source } from "../lib/api";
import { modalityMeta } from "../lib/fileTypes";

function sourceLabel(meta: Record<string, unknown>, index: number): string {
  // `document` = doc_id base (nome original do arquivo). `source` é o nome
  // temporário com UUID, então só entra como último recurso.
  const candidate =
    (meta.document as string) ||
    (meta.file_name as string) ||
    (meta.doc_id as string) ||
    (meta.title as string) ||
    (meta.source as string);
  return candidate || `Trecho ${index + 1}`;
}

function pageLabel(meta: Record<string, unknown>): string | null {
  const page = meta.page ?? meta.page_number ?? meta.sheet;
  return page != null ? `pág. ${page}` : null;
}

export function Sources({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-2.5">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-full border border-line bg-paper/70 px-2.5 py-1 text-xs font-semibold text-muted transition hover:border-line-strong hover:text-ink"
      >
        <Quote className="size-3.5 text-accent" />
        {sources.length} {sources.length === 1 ? "fonte" : "fontes"}
        <ChevronDown
          className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <ul className="mt-2 space-y-2 [animation:var(--animate-fade)]">
          {sources.map((src, i) => {
            const meta = modalityMeta(src.metadata?.modality as string | undefined);
            const Icon = meta.icon;
            const page = pageLabel(src.metadata ?? {});
            return (
              <li
                key={i}
                className="rounded-xl border border-line bg-paper/80 p-3 text-sm"
              >
                <div className="mb-1.5 flex items-center gap-2">
                  <span className={`grid size-6 shrink-0 place-items-center rounded-lg ${meta.bg} ${meta.fg}`}>
                    <Icon className="size-3.5" strokeWidth={2.4} />
                  </span>
                  <span className="min-w-0 flex-1 truncate text-xs font-semibold text-ink">
                    {sourceLabel(src.metadata ?? {}, i)}
                  </span>
                  {page && (
                    <span className="shrink-0 rounded-full bg-canvas px-1.5 py-0.5 text-[10px] font-medium text-muted">
                      {page}
                    </span>
                  )}
                  {typeof src.score === "number" && (
                    <span
                      className="shrink-0 font-mono text-[10px] text-faint"
                      title="Similaridade"
                    >
                      {src.score.toFixed(3)}
                    </span>
                  )}
                </div>
                <p className="line-clamp-4 whitespace-pre-wrap text-xs leading-relaxed text-muted">
                  {src.text}
                </p>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
