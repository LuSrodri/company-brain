import { CheckCircle2, Info, X, XCircle } from "lucide-react";
import { useToast, type ToastKind } from "../hooks/useToast.tsx";

const ICONS: Record<ToastKind, typeof Info> = {
  success: CheckCircle2,
  error: XCircle,
  info: Info,
};

const ACCENT: Record<ToastKind, string> = {
  success: "text-jade",
  error: "text-accent-deep",
  info: "text-mod-document",
};

export function Toaster() {
  const { toasts, dismiss } = useToast();

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[min(92vw,22rem)] flex-col gap-2">
      {toasts.map((t) => {
        const Icon = ICONS[t.kind];
        return (
          <div
            key={t.id}
            role="status"
            className="pointer-events-auto flex items-start gap-3 rounded-2xl border border-line bg-paper/95 p-3.5 shadow-lift backdrop-blur [animation:var(--animate-pop)]"
          >
            <Icon className={`mt-0.5 size-5 shrink-0 ${ACCENT[t.kind]}`} strokeWidth={2.2} />
            <p className="flex-1 text-sm leading-snug text-ink">{t.message}</p>
            <button
              onClick={() => dismiss(t.id)}
              className="-m-1 rounded-lg p-1 text-faint transition hover:bg-canvas hover:text-ink"
              aria-label="Dispensar notificação"
            >
              <X className="size-4" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
