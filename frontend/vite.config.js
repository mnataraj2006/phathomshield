import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendUrl = env.VITE_API_URL || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      // Dev-only proxy: forwards API calls to the local backend.
      // In production, VITE_API_URL is set to the deployed backend URL
      // and all components call it directly (no proxy needed).
      proxy: {
        '/validate':       { target: backendUrl, changeOrigin: true },
        '/recover':        { target: backendUrl, changeOrigin: true },
        '/result':         { target: backendUrl, changeOrigin: true },
        '/health':         { target: backendUrl, changeOrigin: true },
        '/activity':       { target: backendUrl, changeOrigin: true },
        '/stats':          { target: backendUrl, changeOrigin: true },
        '/cases':          { target: backendUrl, changeOrigin: true },
        '/images':         { target: backendUrl, changeOrigin: true },
        '/radiology-report': { target: backendUrl, changeOrigin: true },
      },
    },
  }
})
