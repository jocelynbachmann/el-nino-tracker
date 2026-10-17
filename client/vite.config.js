import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
    plugins: [react(), tailwindcss()],
    resolve: {
        alias: {
            "@": fileURLToPath(new URL("./src", import.meta.url)),
        },
    },
    server: {
        proxy: {
            // /ocean/maps/temperature/2026-09 -> http://localhost:5001/maps/temperature/2026-09
            "/ocean": {
                target: "http://localhost:5001",
                changeOrigin: true,
                rewrite: (path) => path.replace(/^\/ocean/, ""),
            },
        },
    },
});
