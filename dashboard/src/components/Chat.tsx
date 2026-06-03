import { useRef } from "react";
import { Menu, Trash2 } from "lucide-react";
import { useChat } from "../hooks/useChat";
import { MessageList } from "./MessageList";
import { EmptyState } from "./EmptyState";
import { Composer, type ComposerHandle } from "./Composer";

export function Chat({ onOpenMenu }: { onOpenMenu: () => void }) {
  const { messages, pending, send, clear } = useChat();
  const composerRef = useRef<ComposerHandle>(null);

  const hasMessages = messages.length > 0;

  return (
    <main className="flex h-full min-w-0 flex-1 flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-line bg-canvas/70 px-4 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-2">
          <button
            onClick={onOpenMenu}
            className="rounded-xl p-2 text-muted transition hover:bg-paper hover:text-ink md:hidden"
            aria-label="Abrir menu"
          >
            <Menu className="size-5" />
          </button>
          <div className="leading-tight">
            <h1 className="font-display text-base font-semibold tracking-tight text-ink">
              Chat
            </h1>
            <p className="hidden text-xs text-muted sm:block">
              Respostas com fontes, fundamentadas na sua base
            </p>
          </div>
        </div>

        {hasMessages && (
          <button
            onClick={clear}
            className="inline-flex items-center gap-1.5 rounded-xl border border-line bg-paper/70 px-2.5 py-1.5 text-xs font-medium text-muted transition hover:border-accent-soft hover:text-accent-deep"
            title="Limpar conversa"
          >
            <Trash2 className="size-3.5" />
            <span className="hidden sm:inline">Limpar</span>
          </button>
        )}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {hasMessages ? (
          <MessageList messages={messages} pending={pending} />
        ) : (
          <EmptyState onPick={(text) => composerRef.current?.fill(text)} />
        )}
      </div>

      <Composer ref={composerRef} disabled={pending} onSend={(text) => send(text)} />
    </main>
  );
}
