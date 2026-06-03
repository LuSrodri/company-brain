import { AlertTriangle, Sparkles } from "lucide-react";
import type { ChatMessage } from "../hooks/useChat";
import { Markdown } from "./Markdown";
import { Sources } from "./Sources";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end [animation:var(--animate-rise)]">
        <div className="max-w-[min(46rem,85%)] rounded-3xl rounded-br-lg bg-ink px-4 py-2.5 text-[0.94rem] leading-relaxed text-paper shadow-soft">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3 [animation:var(--animate-rise)]">
      <span
        className={`mt-0.5 grid size-8 shrink-0 place-items-center rounded-2xl shadow-soft ${
          message.failed ? "bg-accent-soft text-accent-deep" : "bg-ink text-accent"
        }`}
      >
        {message.failed ? (
          <AlertTriangle className="size-4" />
        ) : (
          <Sparkles className="size-4" strokeWidth={2.2} />
        )}
      </span>

      <div className="min-w-0 max-w-[min(46rem,85%)]">
        <div
          className={`rounded-3xl rounded-tl-lg border px-4 py-3 shadow-soft ${
            message.failed
              ? "border-accent-soft bg-accent-soft/40"
              : "border-line bg-paper"
          }`}
        >
          {message.failed ? (
            <p className="text-sm text-accent-deep">{message.content}</p>
          ) : (
            <Markdown>{message.content}</Markdown>
          )}
        </div>
        {message.sources && <Sources sources={message.sources} />}
      </div>
    </div>
  );
}
