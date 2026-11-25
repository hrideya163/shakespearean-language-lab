import { defineConfig } from "vite";

// During development, forward /api requests to the FastAPI server on port 8000.
export default defineConfig({
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
