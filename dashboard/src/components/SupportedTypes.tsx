import { MODALITIES } from "../lib/fileTypes";

/** Chips com as modalidades suportadas pela ingestão. */
export function SupportedTypes() {
  return (
    <div className="flex flex-wrap gap-1.5">
      {MODALITIES.map((m) => {
        const Icon = m.icon;
        return (
          <span
            key={m.id}
            className={`inline-flex items-center gap-1.5 rounded-full ${m.bg} px-2.5 py-1 text-xs font-semibold ${m.fg}`}
            title={m.exts.join("  ")}
          >
            <Icon className="size-3.5" strokeWidth={2.4} />
            {m.label}
          </span>
        );
      })}
    </div>
  );
}
