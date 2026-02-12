import { Routes, Route } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { ModelComparison } from './pages/ModelComparison';
import { ErrorDistributions } from './pages/ErrorDistributions';
import { StormPerformance } from './pages/StormPerformance';
import { StormTracker } from './pages/StormTracker';
import { Methodology } from './pages/Methodology';
import { loadData } from './lib/data';
import type { DashboardData } from './types';
import { Loader2 } from 'lucide-react';

function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="bg-white rounded-xl p-8 shadow-lg border border-red-200 max-w-md">
          <h2 className="text-xl font-semibold text-red-800 mb-2">Failed to load data</h2>
          <p className="text-slate-600 text-sm mb-4">{error}</p>
          <p className="text-slate-500 text-sm">
            Copy the CSV files from <code className="bg-slate-200 px-1 rounded">streamboard/data/</code> into{' '}
            <code className="bg-slate-200 px-1 rounded">dashboard/public/data/</code>
          </p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-12 h-12 text-indigo-600 animate-spin" />
          <p className="text-slate-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<ModelComparison data={data} />} />
        <Route path="errors" element={<ErrorDistributions data={data} />} />
        <Route path="storms" element={<StormPerformance data={data} />} />
        <Route path="tracker" element={<StormTracker data={data} />} />
        <Route path="methodology" element={<Methodology />} />
      </Route>
    </Routes>
  );
}

export default App;
