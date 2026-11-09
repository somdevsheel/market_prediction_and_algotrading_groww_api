import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: `npm run dev` on :5173 proxies API calls to FastAPI on :8100.
// Prod: `npm run build` -> dist/, served by FastAPI at "/".
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8100", "/metrics": "http://127.0.0.1:8100" },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
