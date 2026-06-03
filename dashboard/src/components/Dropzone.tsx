import { useId, useRef, useState, type DragEvent } from "react";
import { Loader2, UploadCloud } from "lucide-react";
import { ACCEPT_ATTR } from "../lib/fileTypes";
import type { UploadJob } from "../hooks/useDocuments";

export function Dropzone({
  onFiles,
  uploads,
}: {
  onFiles: (files: FileList | File[]) => void;
  uploads: UploadJob[];
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();
  const [dragging, setDragging] = useState(false);
  const busy = uploads.some((u) => u.status === "uploading");

  function handleDrop(e: DragEvent<HTMLLabelElement>) {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) onFiles(e.dataTransfer.files);
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`group relative flex cursor-pointer flex-col items-center gap-2 overflow-hidden rounded-2xl border-2 border-dashed px-4 py-6 text-center transition-all duration-200 ${
          dragging
            ? "border-accent bg-accent-soft/60 scale-[1.01]"
            : "border-line-strong bg-paper/60 hover:border-accent/60 hover:bg-paper"
        }`}
      >
        <span className="dot-grid pointer-events-none absolute inset-0 opacity-40 [mask-image:radial-gradient(circle_at_center,black,transparent_75%)]" />
        <span
          className={`relative grid size-11 place-items-center rounded-2xl transition-colors ${
            dragging ? "bg-accent text-paper" : "bg-canvas text-accent group-hover:bg-accent-soft"
          }`}
        >
          {busy ? (
            <Loader2 className="size-5 animate-spin" />
          ) : (
            <UploadCloud className="size-5" strokeWidth={2.2} />
          )}
        </span>
        <span className="relative text-sm font-semibold text-ink">
          {dragging ? "Solte para enviar" : "Adicionar documentos"}
        </span>
        <span className="relative text-xs text-muted">
          Arraste e solte ou{" "}
          <span className="font-semibold text-accent-deep underline underline-offset-2">
            escolha arquivos
          </span>
        </span>
        <input
          id={inputId}
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT_ATTR}
          className="sr-only"
          onChange={(e) => {
            if (e.target.files?.length) onFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </label>

      {uploads.length > 0 && (
        <ul className="mt-2 space-y-1.5">
          {uploads.map((job) => (
            <li
              key={job.id}
              className="flex items-center gap-2 rounded-xl border border-line bg-paper px-2.5 py-1.5 text-xs [animation:var(--animate-rise)]"
            >
              {job.status === "uploading" && (
                <Loader2 className="size-3.5 shrink-0 animate-spin text-accent" />
              )}
              {job.status === "done" && (
                <span className="size-2 shrink-0 rounded-full bg-jade" />
              )}
              {job.status === "error" && (
                <span className="size-2 shrink-0 rounded-full bg-accent" />
              )}
              <span className="flex-1 truncate font-medium text-ink">{job.name}</span>
              <span className="text-faint">
                {job.status === "uploading"
                  ? "enviando…"
                  : job.status === "done"
                    ? "pronto"
                    : "erro"}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
