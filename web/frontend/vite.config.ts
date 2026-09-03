import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 개발 서버(npm run dev)는 web 백엔드로 프록시한다 — 프론트만 따로 띄워도
// /api와 /ws가 그대로 동작하게 해서 컨테이너 재빌드 없이 UI를 고칠 수 있다.
const backend = process.env.BACKEND_URL ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': backend,
      '/ws': { target: backend.replace('http', 'ws'), ws: true },
    },
  },
  build: { outDir: 'dist' },
})
