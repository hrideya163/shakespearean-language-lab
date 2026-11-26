import { defineConfig } from "vite";

// During development, forward /api (and the experiment figures in /results) to the FastAPI server on port 8000.
export default defineConfig({
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/results": "http://localhost:8000",
    },
  },
});
