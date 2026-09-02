import { defineConfig } from "vite";
import { fileURLToPath, URL } from "node:url";
import process from "node:process"; // explicit import so ESLint's browser globals still apply here
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import basicSsl from '@vitejs/plugin-basic-ssl';

// The realtime backend (task C) runs on :8000 by default. We proxy /health and
// /ws through the dev server so the browser speaks to it same-origin — no CORS,
// and the WebSocket URL stays relative. Override the target with BACKEND_ORIGIN.
const BACKEND = process.env.BACKEND_ORIGIN || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss(), basicSsl()],
  // "@" points at /src, so imports stay stable no matter how deep a file sits
  // (e.g. import { pct } from "@/lib/contract.js"). See README → "Import paths".
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    proxy: {
      "/health": { target: BACKEND, changeOrigin: true },
      "/ws": { target: BACKEND, changeOrigin: true, ws: true },
      "/predict-file": { target: BACKEND, changeOrigin: true },
      "/narrate": { target: BACKEND, changeOrigin: true },
    },
  },
});
