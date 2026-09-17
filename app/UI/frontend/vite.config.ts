import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output goes to UI/static, which UI/main.py serves. During `npm run dev`
// the API is proxied to the FastAPI service on :4002.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../static", emptyOutDir: true },
  server: { proxy: { "/v1": "http://127.0.0.1:4002" } },
});
