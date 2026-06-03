import {
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { ArrowUp } from "lucide-react";

export interface ComposerHandle {
  fill: (text: string) => void;
}

export function Composer({
  onSend,
  disabled,
  ref,
}: {
  onSend: (text: string) => void;
  disabled: boolean;
  ref?: React.Ref<ComposerHandle>;
}) {
  const [value, setValue] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  function autosize() {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "0px";
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }

  useEffect(autosize, [value]);

  useImperativeHandle(ref, () => ({
    fill: (text: string) => {
      setValue(text);
      requestAnimationFrame(() => taRef.current?.focus());
    },
  }));

  function submit() {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
  }

  function handleKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <div className="border-t border-line bg-canvas/70 px-4 py-3 backdrop-blur-sm">
      <div className="mx-auto flex w-full max-w-3xl items-end gap-2 rounded-3xl border border-line-strong bg-paper p-2 shadow-soft transition-colors focus-within:border-accent/60">
        <textarea
          ref={taRef}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Pergunte algo sobre os seus documentos…"
          className="max-h-50 flex-1 resize-none bg-transparent px-3 py-2 text-[0.95rem] leading-relaxed text-ink placeholder:text-faint focus:outline-none"
        />
        <button
          onClick={submit}
          disabled={disabled || value.trim().length === 0}
          className="grid size-10 shrink-0 place-items-center rounded-2xl bg-accent text-paper shadow-soft transition-all hover:bg-accent-deep disabled:cursor-not-allowed disabled:bg-line-strong disabled:text-faint disabled:shadow-none"
          aria-label="Enviar mensagem"
        >
          <ArrowUp className="size-5" strokeWidth={2.4} />
        </button>
      </div>
      <p className="mx-auto mt-1.5 max-w-3xl px-2 text-center text-[11px] text-faint">
        <kbd className="font-mono">Enter</kbd> envia ·{" "}
        <kbd className="font-mono">Shift</kbd>+<kbd className="font-mono">Enter</kbd> quebra linha
      </p>
    </div>
  );
}
