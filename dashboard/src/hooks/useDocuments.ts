import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type DocumentInfo } from "../lib/api";
import { isSupported } from "../lib/fileTypes";
import { useToast } from "./useToast.tsx";

export interface UploadJob {
  id: string;
  name: string;
  status: "uploading" | "done" | "error";
  detail?: string;
}

interface DocumentsState {
  documents: DocumentInfo[];
  totalChunks: number;
  loading: boolean;
  error: string | null;
  uploads: UploadJob[];
  refresh: () => Promise<void>;
  upload: (files: FileList | File[]) => Promise<void>;
  remove: (doc: DocumentInfo) => Promise<void>;
}

export function useDocuments(): DocumentsState {
  const { push } = useToast();
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<UploadJob[]>([]);

  const refresh = useCallback(async () => {
    try {
      const res = await api.listDocuments();
      setDocuments(res.documents);
      setTotalChunks(res.total_chunks);
      setError(null);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "Falha ao listar documentos.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      if (list.length === 0) return;

      for (const file of list) {
        if (!isSupported(file.name)) {
          push("error", `Tipo não suportado: ${file.name}`);
          continue;
        }
        const jobId = `${file.name}-${Date.now()}-${Math.random()}`;
        setUploads((prev) => [
          ...prev,
          { id: jobId, name: file.name, status: "uploading" },
        ]);

        try {
          const res = await api.uploadDocument(file);
          const verb = res.status === "unchanged" ? "já estava atualizado" : "indexado";
          setUploads((prev) =>
            prev.map((j) =>
              j.id === jobId ? { ...j, status: "done", detail: res.status } : j,
            ),
          );
          push("success", `${file.name} ${verb}.`);
          await refresh();
        } catch (err) {
          const msg = err instanceof ApiError ? err.message : "Falha no upload.";
          setUploads((prev) =>
            prev.map((j) =>
              j.id === jobId ? { ...j, status: "error", detail: msg } : j,
            ),
          );
          push("error", `${file.name}: ${msg}`);
        } finally {
          // Remove o job da fila visual após um tempo.
          window.setTimeout(() => {
            setUploads((prev) => prev.filter((j) => j.id !== jobId));
          }, 2600);
        }
      }
    },
    [push, refresh],
  );

  const remove = useCallback(
    async (doc: DocumentInfo) => {
      // Atualização otimista.
      const snapshot = documents;
      setDocuments((prev) => prev.filter((d) => d.doc_id !== doc.doc_id));
      try {
        await api.deleteDocument(doc.doc_id);
        push("success", `Documento removido: ${doc.doc_id || doc.source || "Documento"}`);
        await refresh();
      } catch (err) {
        setDocuments(snapshot); // rollback
        const msg = err instanceof ApiError ? err.message : "Falha ao remover.";
        push("error", msg);
      }
    },
    [documents, push, refresh],
  );

  return {
    documents,
    totalChunks,
    loading,
    error,
    uploads,
    refresh,
    upload,
    remove,
  };
}
