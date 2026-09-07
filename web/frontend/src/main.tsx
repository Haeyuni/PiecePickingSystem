import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import ControlPage from './pages/ControlPage'
import HistoryPage from './pages/HistoryPage'
import './styles.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        {/* 두 화면은 상태를 공유하지 않는다 (화면정의서 1절) */}
        <Route path="/" element={<ControlPage />} />
        <Route path="/history" element={<HistoryPage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
