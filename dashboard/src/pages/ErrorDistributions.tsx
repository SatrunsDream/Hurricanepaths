import { useState, useMemo } from 'react';
import { Chart } from '../components/Chart';
import { Info } from 'lucide-react';
import type { DashboardData } from '../types';

const COLORS = { null: '#e74c3c', kf: '#3498db' };

function computeKDE(values: number[], n: number = 200) {
  if (values.length < 2) return { x: [], y: [] };
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const bandwidth = 1.06 * Math.sqrt(values.reduce((s, v) => s + (v - values.reduce((a, b) => a + b, 0) / values.length) ** 2, 0) / values.length) * Math.pow(values.length, -0.2);
  const x = Array.from({ length: n }, (_, i) => min + (i / (n - 1)) * (max - min));
  const y = x.map((xi) => {
    let sum = 0;
    for (const v of values) {
      sum += Math.exp(-0.5 * ((xi - v) / bandwidth) ** 2);
    }
    return sum / (values.length * bandwidth * Math.sqrt(2 * Math.PI));
  });
  return { x, y };
}

export function ErrorDistributions({ data }: { data: DashboardData }) {
  const errors = data.errors;
  const leadTimes = useMemo(() => [...new Set(errors.map((e) => e.lead_time_hours))].sort((a, b) => a - b), [errors]);
  const basins = useMemo(() => ['All', ...([...new Set(errors.map((e) => e.basin).filter(Boolean))].sort() as string[])], [errors]);
  const natures = useMemo(() => ['All', ...([...new Set(errors.map((e) => e.nature).filter(Boolean))].sort() as string[])], [errors]);

  const [leadTime, setLeadTime] = useState<number>(leadTimes[2] ?? leadTimes[0]);
  const [model, setModel] = useState<'Both' | 'Kalman Filter' | 'Null Model'>('Both');
  const [basin, setBasin] = useState('All');
  const [nature, setNature] = useState('All');
  const [logScale, setLogScale] = useState(true);

  const filtered = useMemo(() => {
    let f = errors.filter((e) => e.lead_time_hours === leadTime);
    if (model === 'Kalman Filter') f = f.filter((e) => e.model === 'Kalman Filter');
    else if (model === 'Null Model') f = f.filter((e) => e.model === 'Null Model');
    if (basin !== 'All') f = f.filter((e) => e.basin === basin);
    if (nature !== 'All') f = f.filter((e) => e.nature === nature);
    return f;
  }, [errors, leadTime, model, basin, nature]);

  const kfData = filtered.filter((e) => e.model === 'Kalman Filter').map((e) => e.error_km);
  const nullData = filtered.filter((e) => e.model === 'Null Model').map((e) => e.error_km);

  const histData = [
    {
      x: kfData,
      type: 'histogram',
      name: 'Kalman Filter',
      opacity: 0.6,
      marker: { color: COLORS.kf },
      nbinsx: 50,
      histnorm: 'probability density',
    },
    {
      x: nullData,
      type: 'histogram',
      name: 'Null Model',
      opacity: 0.6,
      marker: { color: COLORS.null },
      nbinsx: 50,
      histnorm: 'probability density',
    },
  ];

  const kfKde = computeKDE(kfData);
  const nullKde = computeKDE(nullData);

  const kdeData = [
    ...histData,
    {
      x: kfKde.x,
      y: kfKde.y,
      type: 'scatter',
      mode: 'lines',
      name: 'KF KDE',
      line: { color: COLORS.kf, width: 2.5, dash: 'dash' },
    },
    {
      x: nullKde.x,
      y: nullKde.y,
      type: 'scatter',
      mode: 'lines',
      name: 'Null KDE',
      line: { color: COLORS.null, width: 2.5, dash: 'dash' },
    },
  ];

  const kfSorted = [...kfData].sort((a, b) => a - b);
  const nullSorted = [...nullData].sort((a, b) => a - b);
  const cumData = [
    {
      x: kfSorted,
      y: kfSorted.map((_, i) => ((i + 1) / kfSorted.length) * 100),
      type: 'scatter',
      mode: 'lines',
      name: 'Kalman Filter',
      line: { color: COLORS.kf, width: 3 },
    },
    {
      x: nullSorted,
      y: nullSorted.map((_, i) => ((i + 1) / nullSorted.length) * 100),
      type: 'scatter',
      mode: 'lines',
      name: 'Null Model',
      line: { color: COLORS.null, width: 3 },
    },
  ].filter((d) => (d.x as number[]).length > 0);

  const boxData = [
    {
      y: logScale ? kfData.filter((v) => v > 0) : kfData,
      type: 'box',
      name: 'Kalman Filter',
      marker: { color: COLORS.kf },
      boxmean: 'sd',
    },
    {
      y: logScale ? nullData.filter((v) => v > 0) : nullData,
      type: 'box',
      name: 'Null Model',
      marker: { color: COLORS.null },
      boxmean: 'sd',
    },
  ];

  const catData = useMemo(() => {
    const byModel = new Map<string, Map<string, number>>();
    filtered.forEach((e) => {
      if (!byModel.has(e.model)) byModel.set(e.model, new Map());
      const m = byModel.get(e.model)!;
      const cat = e.error_category || 'Unknown';
      m.set(cat, (m.get(cat) || 0) + 1);
    });
    const allCats = [...new Set([...byModel.values()].flatMap((m) => [...m.keys()]))].sort();
    const kfPct = allCats.map((cat) => {
      const m = byModel.get('Kalman Filter');
      const total = m ? [...m.values()].reduce((a, b) => a + b, 0) : 1;
      return ((m?.get(cat) || 0) / total) * 100;
    });
    const nullPct = allCats.map((cat) => {
      const m = byModel.get('Null Model');
      const total = m ? [...m.values()].reduce((a, b) => a + b, 0) : 1;
      return ((m?.get(cat) || 0) / total) * 100;
    });
    return [
      { x: allCats, y: kfPct, type: 'bar' as const, name: 'Kalman Filter', marker: { color: COLORS.kf } },
      { x: allCats, y: nullPct, type: 'bar' as const, name: 'Null Model', marker: { color: COLORS.null } },
    ];
  }, [filtered]);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-800 to-indigo-600 bg-clip-text text-transparent">
          Error Distribution Analysis
        </h1>
        <p className="text-slate-600 mt-2">
          Deep dive into forecast error patterns and distributions
        </p>
      </header>

      <div className="flex flex-wrap gap-4">
        <FilterSelect
          label="Lead Time"
          value={leadTime}
          onChange={(v) => setLeadTime(Number(v))}
          options={leadTimes}
        />
        <FilterSelect
          label="Model"
          value={model}
          onChange={(v) => setModel(v as typeof model)}
          options={['Both', 'Kalman Filter', 'Null Model']}
        />
        <FilterSelect label="Basin" value={basin} onChange={setBasin} options={basins} />
        <FilterSelect label="Storm Type" value={nature} onChange={setNature} options={natures} />
      </div>

      <div className="flex items-center gap-2 text-slate-600 text-sm">
        <Info className="w-4 h-4" />
        Showing {filtered.length.toLocaleString()} forecast instances for {leadTime}h lead time
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold mb-4">
          Error Distribution with KDE at {leadTime}h Lead Time
        </h2>
        <Chart data={kdeData} layout={{ barmode: 'overlay', height: 500 }} />
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold mb-4">Cumulative Error Distribution</h2>
        <Chart data={cumData} layout={{ height: 500 }} />
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Error Distribution (Box Plot)</h2>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={logScale}
              onChange={(e) => setLogScale(e.target.checked)}
              className="rounded"
            />
            Use logarithmic scale
          </label>
        </div>
        <Chart
          data={boxData}
          layout={{
            height: 500,
            yaxis: { type: logScale ? 'log' : 'linear', title: 'Forecast Error (km)' },
          }}
        />
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold mb-4">Error Category Breakdown</h2>
        <Chart
          data={catData}
          layout={{
            barmode: 'group',
            xaxis: { title: 'Error Category' },
            yaxis: { title: 'Percentage (%)' },
            height: 400,
          }}
        />
      </div>
    </div>
  );
}

function FilterSelect<T extends string | number>({
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
