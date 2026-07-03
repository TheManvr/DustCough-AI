import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import { copy } from './assets/copy'
import { logos } from './assets/logos'
import HomePage from './pages/HomePage'
import RecordPage from './pages/RecordPage'
import ResultPage from './pages/ResultPage'
import SymptomFormPage from './pages/SymptomFormPage'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <div className="app-wrapper">
        <header className="app-header" aria-label={copy.appHeaderAria}>
          <Link className="brand-home-link" to="/" aria-label="กลับหน้าแรก DustCough AI">
            <img className="brand-logo" src={logos.combination} alt={copy.logoMainAlt} />
          </Link>
        </header>

        <main className="app-container">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/record" element={<RecordPage />} />
            <Route path="/symptoms" element={<SymptomFormPage />} />
            <Route path="/result" element={<ResultPage />} />
          </Routes>
        </main>

        <footer className="app-footer">
          <img className="footer-logo" src={logos.wordmark} alt={copy.footerLogoAlt} />
          <p>{copy.appFooterText}</p>
          <span>{copy.appFooterDisclaimer}</span>
        </footer>
      </div>
    </BrowserRouter>
  )
}

export default App
