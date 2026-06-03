import { useEffect, useRef } from "react";
import type { ChatMessage } from "../hooks/useChat";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";

export function MessageList({
  messages,
  pending,
}: {
  messages: ChatMessage[];
  pending: boolean;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-4 py-6">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      {pending && (
        <div className="flex gap-3">
          <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-2xl bg-ink text-accent shadow-soft">
            <span className="thinking-dot size-2 rounded-full bg-accent" />
          </span>
          <div className="rounded-3xl rounded-tl-lg border border-line bg-paper px-2 shadow-soft">
            <TypingIndicator />
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
