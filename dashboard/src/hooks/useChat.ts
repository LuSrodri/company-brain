import { useCallback, useRef, useState } from "react";
import { api, ApiError, type ChatTurn, type Source } from "../lib/api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  /** Mensagem do assistente que falhou (mostra estado de erro). */
  failed?: boolean;
}

let counter = 0;
const newId = () => `m${++counter}-${Date.now()}`;

interface ChatState {
  messages: ChatMessage[];
  pending: boolean;
  send: (text: string, topK?: number) => Promise<void>;
  clear: () => void;
}

export function useChat(): ChatState {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  const clear = useCallback(() => {
    controllerRef.current?.abort();
    setMessages([]);
    setPending(false);
  }, []);

  const send = useCallback(
    async (text: string, topK?: number) => {
      const trimmed = text.trim();
      if (!trimmed || pending) return;

      const history: ChatTurn[] = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const userMsg: ChatMessage = { id: newId(), role: "user", content: trimmed };
      setMessages((prev) => [...prev, userMsg]);
      setPending(true);

      const controller = new AbortController();
      controllerRef.current = controller;

      try {
        const res = await api.chat(trimmed, history, { topK, signal: controller.signal });
        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            role: "assistant",
            content: res.answer,
            sources: res.sources,
          },
        ]);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        const msg =
          err instanceof ApiError
            ? err.message
            : "Algo deu errado ao consultar a base.";
        setMessages((prev) => [
          ...prev,
          { id: newId(), role: "assistant", content: msg, failed: true },
        ]);
      } finally {
        setPending(false);
        controllerRef.current = null;
      }
    },
    [messages, pending],
  );

  return { messages, pending, send, clear };
}
