/**
 * Espelha `SUPPORTED_EXTS` de api/app/core/ingestion.py e agrupa as extensões
 * por modalidade, com rótulo, ícone (lucide) e classes de cor para a UI.
 */
import {
  FileText,
  Image,
  Music,
  Sheet,
  Type,
  FileType,
  type LucideIcon,
} from "lucide-react";
import type { Modality } from "./api";

export interface ModalityMeta {
  id: Modality;
  label: string;
  /** Extensões aceitas (com ponto), conforme o backend. */
  exts: string[];
  icon: LucideIcon;
  /** Classe utilitária Tailwind para a cor de texto/ícone. */
  fg: string;
  /** Classe utilitária Tailwind para o fundo suave. */
  bg: string;
}

export const MODALITIES: ModalityMeta[] = [
  {
    id: "text",
    label: "Texto",
    exts: [".txt", ".md", ".markdown", ".rst", ".csv", ".json"],
    icon: Type,
    fg: "text-mod-text",
    bg: "bg-mod-text-soft",
  },
  {
    id: "pdf",
    label: "PDF",
    exts: [".pdf"],
    icon: FileText,
    fg: "text-mod-pdf",
    bg: "bg-mod-pdf-soft",
  },
  {
    id: "image",
    label: "Imagem",
    exts: [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"],
    icon: Image,
    fg: "text-mod-image",
    bg: "bg-mod-image-soft",
  },
  {
    id: "audio",
    label: "Áudio",
    exts: [".wav", ".mp3", ".flac", ".ogg", ".m4a"],
    icon: Music,
    fg: "text-mod-audio",
    bg: "bg-mod-audio-soft",
  },
  {
    id: "spreadsheet",
    label: "Planilha",
    exts: [".xlsx", ".xlsm"],
    icon: Sheet,
    fg: "text-mod-spreadsheet",
    bg: "bg-mod-spreadsheet-soft",
  },
  {
    id: "document",
    label: "Documento",
    exts: [".docx"],
    icon: FileType,
    fg: "text-mod-document",
    bg: "bg-mod-document-soft",
  },
];

const BY_ID = new Map(MODALITIES.map((m) => [m.id, m]));

/** Todas as extensões suportadas (achatado). */
export const ALL_EXTS = MODALITIES.flatMap((m) => m.exts);

/** Atributo `accept` para <input type="file">. */
export const ACCEPT_ATTR = ALL_EXTS.join(",");

const EXT_TO_MODALITY = new Map<string, Modality>();
for (const m of MODALITIES) for (const ext of m.exts) EXT_TO_MODALITY.set(ext, m.id);

export function extOf(filename: string): string {
  const i = filename.lastIndexOf(".");
  return i >= 0 ? filename.slice(i).toLowerCase() : "";
}

export function isSupported(filename: string): boolean {
  return EXT_TO_MODALITY.has(extOf(filename));
}

/** Resolve a modalidade a partir de um nome de arquivo. */
export function modalityOfFile(filename: string): ModalityMeta {
  const id = EXT_TO_MODALITY.get(extOf(filename));
  return (id && BY_ID.get(id)) || MODALITIES[0];
}

/** Resolve a meta de modalidade a partir do valor retornado pela API. */
export function modalityMeta(modality: string | null | undefined): ModalityMeta {
  return (modality && BY_ID.get(modality as Modality)) || MODALITIES[0];
}
