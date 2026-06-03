import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// O backend FastAPI (api/) sobe em http://localhost:8000 e NÃO define CORS.
// Em dev fazemos proxy de /api/* -> http://localhost:8000/* (removendo o prefixo),
// evitando qualquer configuração de CORS no backend.
const API_TARGET = process.env.VITE_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
