import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI backend runs on its own origin and does not send CORS headers,
// so the dev server proxies /chat for us. The frontend therefore always talks
// to the same-origin /chat path and never needs an absolute backend URL (which
// would also mean shipping configuration into the browser).
const BACKEND_ORIGIN = 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': {
        target: BACKEND_ORIGIN,
        changeOrigin: true,
      },
    },
  },
  preview: {
    proxy: {
      '/chat': {
        target: BACKEND_ORIGIN,
        changeOrigin: true,
      },
    },
  },
})
