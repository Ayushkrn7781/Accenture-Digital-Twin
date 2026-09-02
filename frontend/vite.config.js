import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The backend port can be overridden with API_PORT for side-by-side runs.
const target = `http://127.0.0.1:${process.env.API_PORT || 8000}`;
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': { target, changeOrigin: true } } },
});
