export function Brand() {
  return (
    <div className="flex items-center gap-3">
      <span className="relative grid size-10 shrink-0 place-items-center rounded-2xl bg-ink shadow-soft">
        <svg viewBox="0 0 32 32" className="size-6" aria-hidden="true">
          <path
            d="M16 6.5c-3.6 0-6.2 2.3-6.2 5.3 0 1.2.4 2.2 1.1 3-.6.8-1 1.8-1 2.9 0 3 2.6 5.3 6.1 5.3s6.1-2.3 6.1-5.3c0-1.1-.4-2.1-1-2.9.7-.8 1.1-1.8 1.1-3 0-3-2.6-5.3-6.2-5.3Z"
            fill="none"
            stroke="#f4efe6"
            strokeWidth="1.6"
          />
          <path d="M16 8.5v15" stroke="#f4efe6" strokeWidth="1.6" strokeLinecap="round" />
          <circle cx="16" cy="13" r="1.7" fill="#ef5a3a" />
          <circle cx="12.4" cy="18.4" r="1.4" fill="#ef5a3a" />
          <circle cx="19.6" cy="18.4" r="1.4" fill="#ef5a3a" />
        </svg>
        <span className="absolute -right-0.5 -top-0.5 size-2.5 rounded-full bg-accent ring-2 ring-paper" />
      </span>
      <div className="leading-tight">
        <p className="font-display text-lg font-semibold tracking-tight text-ink">
          Company Brain
        </p>
        <p className="text-xs font-medium text-muted">Conhecimento ao seu lado</p>
      </div>
    </div>
  );
}
