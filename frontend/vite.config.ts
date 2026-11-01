import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI backend runs on its own origin and does not send CORS headers,
// so the dev server proxies /chat for us. The frontend therefore always talks
// to the same-origin /chat path and never needs an absolute backend URL (which
// would also mean shipping configuration into the browser).
const BACKEND_ORIGIN = 'http://127.0.0.1:8000'

// The dev server is pinned to the IPv4 loopback address on purpose.
//
// Left at Vite's default (`localhost`), Node resolves that name to `::1` first
// on Windows and binds the socket to IPv6 only. A browser asking for the IPv4
// loopback -- which is what `http://localhost:5173` may resolve to -- then finds
// nothing listening there and shows an empty page. Naming 127.0.0.1 explicitly
// makes the server answer on the one address this project documents, while
// still keeping it on loopback: it is never exposed to the LAN.
const DEV_HOST = '127.0.0.1'

// `strictPort` keeps the documented address honest. Without it a busy 5173
// would silently become 5174 and the URL in the README would lead nowhere; with
// it, a taken port fails loudly instead of serving somewhere unexpected.
const sharedServerOptions = {
  host: DEV_HOST,
  strictPort: true,
  proxy: {
    '/chat': {
      target: BACKEND_ORIGIN,
      changeOrigin: true,
    },
    // The save workflow posts to /save and /save/cancel (see
    // frontend/src/api/chat.ts). Without this entry those two calls would hit the
    // dev server itself and 404, so a proposal could never be confirmed; one
    // prefix covers both routes.
    '/save': {
      target: BACKEND_ORIGIN,
      changeOrigin: true,
    },
  },
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // `vite preview` serves the built bundle and needs the same host, port and
  // proxy behaviour as the dev server, so both share one set of options.
  server: sharedServerOptions,
  preview: sharedServerOptions,
})
