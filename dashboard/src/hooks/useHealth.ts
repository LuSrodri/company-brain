import { useEffect, useState } from "react";
import { api, type HealthResponse } from "../lib/api";

export type HealthStatus = "checking" | "online" | "offline";

interface HealthState {
  status: HealthStatus;
  version: string | null;
}

/** Faz polling do /health para mostrar o estado de conexão da API. */
export function useHealth(intervalMs = 15000): HealthState {
  const [state, setState] = useState<HealthState>({
    status: "checking",
    version: null,
  });

  useEffect(() => {
    let alive = true;
    const controller = new AbortController();

    async function check() {
      try {
        const res: HealthResponse = await api.health(controller.signal);
        if (alive) setState({ status: "online", version: res.version });
      } catch {
        if (alive) setState((s) => ({ status: "offline", version: s.version }));
      }
    }

    check();
    const timer = window.setInterval(check, intervalMs);
    return () => {
      alive = false;
      controller.abort();
      window.clearInterval(timer);
    };
  }, [intervalMs]);

  return state;
}
