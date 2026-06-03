import { X } from "lucide-react";
import { Brand } from "./Brand";
import { Dropzone } from "./Dropzone";
import { SupportedTypes } from "./SupportedTypes";
import { DocumentList } from "./DocumentList";
import { StatusBadge } from "./StatusBadge";
import type { useDocuments } from "../hooks/useDocuments";
import type { HealthStatus } from "../hooks/useHealth";

type Docs = ReturnType<typeof useDocuments>;

export function Sidebar({
  docs,
  health,
  onClose,
}: {
  docs: Docs;
  health: { status: HealthStatus; version: string | null };
  onClose?: () => void;
}) {
  return (
    <aside className="flex h-full w-full flex-col gap-4 border-r border-line bg-canvas/80 p-4 backdrop-blur-sm md:w-[21rem]">
      <div className="flex items-center justify-between">
        <Brand />
        {onClose && (
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-muted transition hover:bg-paper hover:text-ink md:hidden"
            aria-label="Fechar menu"
          >
            <X className="size-5" />
          </button>
        )}
      </div>

      <div className="space-y-2">
        <Dropzone onFiles={docs.upload} uploads={docs.uploads} />
        <SupportedTypes />
      </div>

      <DocumentList
        documents={docs.documents}
        loading={docs.loading}
        error={docs.error}
        totalChunks={docs.totalChunks}
        onDelete={docs.remove}
        onRefresh={docs.refresh}
      />

      <div className="flex items-center justify-between border-t border-line pt-3">
        <StatusBadge status={health.status} version={health.version} />
        <span className="font-mono text-[10px] text-faint">RAG · Gemma 4</span>
      </div>
    </aside>
  );
}
