import type { HealthStatus } from "../hooks/useHealth";

const META: Record<HealthStatus, { label: string; dot: string; text: string }> = {
  checking: { label: "Conectando…", dot: "bg-faint", text: "text-muted" },
  online: { label: "API online", dot: "bg-jade", text: "text-jade" },
  offline: { label: "API offline", dot: "bg-accent", text: "text-accent-deep" },
};

export function StatusBadge({
  status,
  version,
}: {
  status: HealthStatus;
  version: string | null;
}) {
  const m = META[status];
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-line bg-paper/70 px-2.5 py-1 text-xs font-medium"
      title={version ? `Company Brain API v${version}` : undefined}
    >
      <span className="relative flex size-2">
        {status === "online" && (
          <span className={`absolute inline-flex size-full animate-ping rounded-full ${m.dot} opacity-60`} />
        )}
        <span className={`relative inline-flex size-2 rounded-full ${m.dot}`} />
      </span>
      <span className={m.text}>{m.label}</span>
      {version && status === "online" && (
        <span className="font-mono text-[10px] text-faint">v{version}</span>
      )}
    </span>
  );
}
