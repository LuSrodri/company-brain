import { useEffect, useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { Chat } from "./components/Chat";
import { Toaster } from "./components/Toaster";
import { useDocuments } from "./hooks/useDocuments";
import { useHealth } from "./hooks/useHealth";

export function App() {
  const docs = useDocuments();
  const health = useHealth();
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Fecha o drawer ao apertar Esc.
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  return (
    <div className="flex h-dvh overflow-hidden">
      {/* Sidebar fixa (desktop) */}
      <div className="hidden md:flex md:shrink-0">
        <Sidebar docs={docs} health={health} />
      </div>

      {/* Drawer (mobile) */}
      {drawerOpen && (
        <div className="fixed inset-0 z-40 md:hidden">
          <div
            className="absolute inset-0 bg-ink/30 backdrop-blur-[2px] [animation:var(--animate-fade)]"
            onClick={() => setDrawerOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-[min(21rem,88vw)] shadow-lift [animation:var(--animate-rise)]">
            <Sidebar docs={docs} health={health} onClose={() => setDrawerOpen(false)} />
          </div>
        </div>
      )}

      <Chat onOpenMenu={() => setDrawerOpen(true)} />

      <Toaster />
    </div>
  );
}
