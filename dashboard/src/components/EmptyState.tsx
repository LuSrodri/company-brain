import { Sparkles } from "lucide-react";

const SUGGESTIONS = [
  "Resuma os principais achados do último relatório.",
  "Quais políticas internas mencionam férias?",
  "O que foi discutido sobre segurança nos documentos?",
  "Liste os pontos de ação do material que enviei.",
];

export function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="flex min-h-full flex-col items-center justify-center px-6 py-10 text-center">
      <span className="mb-5 grid size-16 place-items-center rounded-3xl bg-ink text-accent shadow-lift [animation:var(--animate-pop)]">
        <Sparkles className="size-7" strokeWidth={2} />
      </span>

      <h1 className="font-display text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
        Pergunte ao cérebro da empresa
      </h1>
      <p className="mt-3 max-w-md text-balance text-muted">
        Respostas fundamentadas nos seus documentos — PDFs, planilhas, áudios,
        imagens e mais. Cada resposta vem com as fontes consultadas.
      </p>

      <div className="mt-8 grid w-full max-w-xl gap-2.5 sm:grid-cols-2">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="group rounded-2xl border border-line bg-paper/70 px-4 py-3 text-left text-sm text-ink shadow-soft transition-all hover:-translate-y-0.5 hover:border-accent/50 hover:shadow-lift [animation:var(--animate-rise)]"
            style={{ animationDelay: `${i * 0.06}s` }}
          >
            <span className="line-clamp-2 font-medium">{s}</span>
            <span className="mt-1 inline-block text-xs font-semibold text-accent opacity-0 transition group-hover:opacity-100">
              Perguntar →
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
