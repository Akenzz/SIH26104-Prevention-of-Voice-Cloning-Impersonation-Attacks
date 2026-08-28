import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import LiveMonitor from './pages/LiveMonitor';
import FileAnalysis from './pages/FileAnalysis';
import Settings from './pages/Settings';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="live-monitor" element={<LiveMonitor />} />
        <Route path="file-analysis" element={<FileAnalysis />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}

export default App;
