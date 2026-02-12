import { useState, useMemo } from 'react';
import { Chart } from '../components/Chart';
import { Info } from 'lucide-react';
import type { DashboardData } from '../types';

const COLORS = { null: '#e74c3c', kf: '#3498db' };

export function StormTracker({ data }: { data: DashboardData }) {
  const { comparison, metadata } = data;
  const storms = useMemo(() => [...new Set(comparison.map((c) => c.sid))].sort(), [comparison]);

  const [selectedStorm, setSelectedStorm] = useState(storms[0] ?? '');

  const stormData = useMemo(() => {
    const filtered = comparison.filter((c) => c.sid === selectedStorm);
    const byLead = new Map<number, { kf: number[]; null: number[] }>();
    filtered.forEach((r) => {
      if (!byLead.has(r.lead_time_hours)) byLead.set(r.lead_time_hours, { kf: [], null: [] });
      const v = byLead.get(r.lead_time_hours)!;
      v.kf.push(r.kf_error_km);
      v.null.push(r.null_error_km);
    });
    return [...byLead.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([lead, v]) => ({
        lead_time_hours: lead,
        kf_mean: v.kf.reduce((a, b) => a + b, 0) / v.kf.length,
        null_mean: v.null.reduce((a, b) => a + b, 0) / v.null.length,
      }));
  }, [comparison, selectedStorm]);

  const meta = useMemo(
    () => metadata.find((m) => m.sid === selectedStorm),
    [metadata, selectedStorm]
  );

  const trendData = [
    {
      x: stormData.map((r) => r.lead_time_hours),
      y: stormData.map((r) => r.kf_mean),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Kalman Filter',
      line: { color: COLORS.kf, width: 3 },
      marker: { size: 10 },
    },
    {
      x: stormData.map((r) => r.lead_time_hours),
      y: stormData.map((r) => r.null_mean),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Null Model',
      line: { color: COLORS.null, width: 3 },
      marker: { size: 10 },
    },
  ];

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-800 to-indigo-600 bg-clip-text text-transparent">
          Individual Storm Tracker
        </h1>
        <p className="text-slate-600 mt-2">
          Visualize specific storm tracks and forecast comparisons
        </p>
      </header>

      <div className="flex flex-wrap gap-4 items-end">
        <div>
          <label className="block text-sm font-medium text-slate-600 mb-1">Select Storm</label>
          <select
            value={selectedStorm}
            onChange={(e) => setSelectedStorm(e.target.value)}
            className="rounded-lg border border-slate-300 px-4 py-2 min-w-[200px] focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            {storms.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        {meta?.basin_name && (
          <div className="flex items-center gap-2 text-slate-600 text-sm bg-slate-100 px-4 py-2 rounded-lg">
            <Info className="w-4 h-4" />
            Basin: {meta.basin_name}
          </div>
        )}
      </div>

      {stormData.length === 0 ? (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 text-amber-800">
          No forecast data available for this storm.
        </div>
      ) : (
        <>
          <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
            <h2 className="text-xl font-semibold mb-4">
              Forecast Error Trend for Storm {selectedStorm}
            </h2>
            <Chart
              data={trendData}
              layout={{
                xaxis: { title: 'Lead Time (hours)' },
                yaxis: { title: 'Mean Error (km)' },
                height: 400,
              }}
            />
          </div>

          <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200 overflow-x-auto">
            <h2 className="text-xl font-semibold mb-4">Error Metrics by Lead Time</h2>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200">
                  <th className="text-left py-3 px-4">Lead Time (h)</th>
                  <th className="text-right py-3 px-4">KF Error (km)</th>
                  <th className="text-right py-3 px-4">Null Error (km)</th>
                  <th className="text-right py-3 px-4">Difference (km)</th>
                </tr>
              </thead>
              <tbody>
                {stormData.map((r) => (
                  <tr key={r.lead_time_hours} className="border-b border-slate-100 hover:bg-slate-50">
                    <td className="py-3 px-4">{r.lead_time_hours}</td>
                    <td className="text-right py-3 px-4">{r.kf_mean.toFixed(2)}</td>
                    <td className="text-right py-3 px-4">{r.null_mean.toFixed(2)}</td>
                    <td className="text-right py-3 px-4">
                      {(r.null_mean - r.kf_mean).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-slate-600 text-sm">
        <Info className="w-4 h-4 inline mr-2" />
        Full track visualization with map overlay requires loading the full track dataset (
        <code className="bg-slate-200 px-1 rounded">hurricane_paths_processed.pkl</code>).
      </div>
    </div>
  );
}
