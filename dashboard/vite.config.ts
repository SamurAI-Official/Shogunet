import { defineConfig } from "vite";
import solid from "vite-plugin-solid";

// Build straight into the Python package so `pip wheel shugonet` ships the
// compiled console; dashboard.py serves this directory as static files.
export default defineConfig({
  plugins: [solid()],
  base: "/",
  build: {
    outDir: "../shugonet_web/static",
    emptyOutDir: true,
    target: "es2020",
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:9002",
    },
  },
});
