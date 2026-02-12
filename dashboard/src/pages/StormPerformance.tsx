import { useState, useMemo } from 'react';
import { Chart } from '../components/Chart';
import { Info } from 'lucide-react';
import type { DashboardData } from '../types';

const COLORS = { null: '#e74c3c', kf: '#3498db' };

export function StormPerformance({ data }: { data: DashboardData }) {
  const storms = data.storms;
  const basins = useMemo(() => ['All', ...([...new Set(storms.map((s) => s.basin).filter(Boolean))].sort() as string[])], [storms]);
  const natures = useMemo(() => ['All', ...([...new Set(storms.map((s) => s.nature).filter(Boolean))].sort() as string[])], [storms]);

  const [basin, setBasin] = useState('All');
  const [nature, setNature] = useState('All');
  const [errorThreshold, setErrorThreshold] = useState(
    Math.max(...storms.map((s) => s.kf_mean_error), 500)
  );

  const filtered = useMemo(() => {
    let f = storms.filter((s) => s.kf_mean_error <= errorThreshold);
    if (basin !== 'All') f = f.filter((s) => s.basin === basin);
    if (nature !== 'All') f = f.filter((s) => s.nature === nature);
    return f;
  }, [storms, basin, nature, errorThreshold]);

  const kfBetter = filtered.filter((s) => s.better_model === 'Kalman Filter');
  const nullBetter = filtered.filter((s) => s.better_model === 'Null Model');

  const scatterData = [
    {
      x: kfBetter.map((s) => s.kf_mean_error),
      y: kfBetter.map((s) => s.null_mean_error),
      type: 'scatter',
      mode: 'markers',
      name: 'KF Better',
      marker: { color: COLORS.kf, size: 6, opacity: 0.6 },
      text: kfBetter.map((s) => s.sid),
      hovertemplate: '<b>%{text}</b><br>KF: %{x:.1f} km<br>Null: %{y:.1f} km<extra></extra>',
    },
    {
      x: nullBetter.map((s) => s.kf_mean_error),
      y: nullBetter.map((s) => s.null_mean_error),
      type: 'scatter',
      mode: 'markers',
      name: 'Null Better',
      marker: { color: COLORS.null, size: 6, opacity: 0.6 },
      text: nullBetter.map((s) => s.sid),
      hovertemplate: '<b>%{text}</b><br>KF: %{x:.1f} km<br>Null: %{y:.1f} km<extra></extra>',
    },
    {
      x: [0, Math.max(...filtered.map((s) => Math.max(s.kf_mean_error, s.null_mean_error)), 1)],
      y: [0, Math.max(...filtered.map((s) => Math.max(s.kf_mean_error, s.null_mean_error)), 1)],
      type: 'scatter',
      mode: 'lines',
      name: 'Equal Performance',
      line: { color: 'gray', dash: 'dash', width: 2 },
    },
  ];

  const pieData = [
    {
      values: [nullBetter.length, kfBetter.length],
      labels: ['Null Model', 'Kalman Filter'],
      type: 'pie',
      hole: 0.4,
      marker: { colors: [COLORS.null, COLORS.kf] },
    },
  ];

  const basinPerf = useMemo(() => {
    const byBasin = new Map<string, { kf: number[]; null: number[] }>();
    filtered.forEach((s) => {
      const b = s.basin || 'Unknown';
      if (!byBasin.has(b)) byBasin.set(b, { kf: [], null: [] });
      const v = byBasin.get(b)!;
      v.kf.push(s.kf_mean_error);
      v.null.push(s.null_mean_error);
    });
    return [...byBasin.entries()].map(([basinName, v]) => ({
      basin: basinName,
      kf_mean: v.kf.reduce((a, b) => a + b, 0) / v.kf.length,
      null_mean: v.null.reduce((a, b) => a + b, 0) / v.null.length,
    }));
  }, [filtered]);

  const basinData = [
    {
      x: basinPerf.map((b) => b.basin),
      y: basinPerf.map((b) => b.kf_mean),
      type: 'bar',
      name: 'Kalman Filter',
      marker: { color: COLORS.kf },
    },
    {
      x: basinPerf.map((b) => b.basin),
      y: basinPerf.map((b) => b.null_mean),
      type: 'bar',
      name: 'Null Model',
      marker: { color: COLORS.null },
    },
  ];

  const topStorms = [...filtered]
    .sort((a, b) => b.kf_mean_error - a.kf_mean_error)
    .slice(0, 20);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-800 to-indigo-600 bg-clip-text text-transparent">
          Storm-Level Performance
        </h1>
        <p className="text-slate-600 mt-2">
          Explore which storms each model handles better
        </p>
      </header>

      <div className="flex flex-wrap gap-4 items-end">
        <FilterSelect label="Basin" value={basin} onChange={setBasin} options={basins} />
        <FilterSelect label="Storm Type" value={nature} onChange={setNature} options={natures} />
        <div>
          <label className="block text-sm font-medium text-slate-600 mb-1">
            Max KF Error (km)
          </label>
          <input
            type="range"
            min={Math.min(...storms.map((s) => s.kf_mean_error))}
            max={Math.max(...storms.map((s) => s.kf_mean_error))}
            value={errorThreshold}
            onChange={(e) => setErrorThreshold(parseFloat(e.target.value))}
            className="w-48"
          />
          <span className="ml-2 text-sm">{Math.round(errorThreshold)} km</span>
        </div>
      </div>

      <div className="flex items-center gap-2 text-slate-600 text-sm">
        <Info className="w-4 h-4" />
        Showing {filtered.length.toLocaleString()} storms
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold mb-4">Storm Performance Comparison</h2>
        <Chart
          data={scatterData}
          layout={{
            xaxis: { title: 'Kalman Filter Mean Error (km)' },
            yaxis: { title: 'Null Model Mean Error (km)' },
            height: 600,
          }}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-4">Which Model Performs Better?</h2>
          <Chart data={pieData} layout={{ height: 400 }} />
        </div>
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-4">Average Error by Basin</h2>
          <Chart
            data={basinData}
            layout={{
              barmode: 'group',
              xaxis: { title: 'Basin' },
              yaxis: { title: 'Mean Error (km)' },
              height: 400,
            }}
          />
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200 overflow-x-auto">
        <h2 className="text-xl font-semibold mb-4">Storm Performance Ranking</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="text-left py-3 px-4">Storm ID</th>
              <th className="text-left py-3 px-4">Basin</th>
              <th className="text-left py-3 px-4">Type</th>
              <th className="text-right py-3 px-4">KF Error (km)</th>
              <th className="text-right py-3 px-4">Null Error (km)</th>
              <th className="text-left py-3 px-4">Better Model</th>
            </tr>
          </thead>
          <tbody>
            {topStorms.map((s) => (
              <tr key={s.sid} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="py-3 px-4">{s.sid}</td>
                <td className="py-3 px-4">{s.basin || '-'}</td>
                <td className="py-3 px-4">{s.nature || '-'}</td>
                <td className="text-right py-3 px-4">{s.kf_mean_error.toFixed(2)}</td>
                <td className="text-right py-3 px-4">{s.null_mean_error.toFixed(2)}</td>
                <td className="py-3 px-4">{s.better_model}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FilterSelect<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: T[];
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-600 mb-1">{label}</label>
      <select
        value={String(value)}
        onChange={(e) => onChange(e.target.value as T)}
        className="rounded-lg border border-slate-300 px-4 py-2 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
      >
        {options.map((o) => (
          <option key={String(o)} value={String(o)}>
            {String(o)}
          </option>
        ))}
      </select>
    </div>
  );
}
