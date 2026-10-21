import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, forward API calls to the FastAPI server. In production
// FastAPI serves the built app itself, so the same relative paths work.
const api = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": api,
      "/documents": api,
      "/ingest": api,
      "/chat": api, // also /chat/<thread_id> for history
    },
  },
});
