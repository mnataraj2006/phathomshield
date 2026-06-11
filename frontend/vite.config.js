import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendUrl = env.VITE_API_URL?.replace('/api', '') || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      // Dev proxy: mirrors the Vercel /api/* → Render rewrite so local dev
      // works identically to production without any CORS issues.
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
        // Legacy direct paths still work for backwards compat
        '/validate':         { target: backendUrl, changeOrigin: true },
        '/recover':          { target: backendUrl, changeOrigin: true },
        '/result':           { target: backendUrl, changeOrigin: true },
        '/health':           { target: backendUrl, changeOrigin: true },
        '/activity':         { target: backendUrl, changeOrigin: true },
        '/stats':            { target: backendUrl, changeOrigin: true },
        '/cases':            { target: backendUrl, changeOrigin: true },
        '/images':           { target: backendUrl, changeOrigin: true },
        '/radiology-report': { target: backendUrl, changeOrigin: true },
      },
    },
  }
})
