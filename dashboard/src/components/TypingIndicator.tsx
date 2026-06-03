export function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-2" aria-label="Assistente pensando">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="thinking-dot size-2 rounded-full bg-accent"
          style={{ animationDelay: `${i * 0.18}s` }}
        />
      ))}
      <span className="ml-1 text-xs text-muted">consultando a base…</span>
    </div>
  );
}
