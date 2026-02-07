import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // ==========================================
    // Proxy Configuration
    // ==========================================
    // This allows the frontend (running on port 5173) to forward API requests
    // to the backend (running on port 5000) without CORS issues during development.
    proxy: {
      '/api': {
        target: 'http://localhost:5000', // The backend URL
        changeOrigin: true,              // Needed for virtual hosted sites
        // Rewrite the path: '/api/summarize' becomes '/summarize' on the backend
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
