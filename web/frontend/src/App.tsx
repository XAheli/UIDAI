/**
 * Main App Component
 * Author: Shuvam Banerji Seal's Team
 */

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Header, Footer } from './components';
import {
  Dashboard,
  TimeSeriesPage,
  GeographicPage,
  DemographicPage,
  StatisticalPage,
  MLPage,
  DownloadsPage,
} from './pages';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/time-series" element={<TimeSeriesPage />} />
            <Route path="/geographic" element={<GeographicPage />} />
            <Route path="/demographic" element={<DemographicPage />} />
            <Route path="/statistical" element={<StatisticalPage />} />
            <Route path="/ml" element={<MLPage />} />
            <Route path="/downloads" element={<DownloadsPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
