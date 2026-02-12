import { useMemo } from 'react';
import { Chart } from '../components/Chart';
import { AlertCircle, TrendingUp, Target } from 'lucide-react';
import type { DashboardData } from '../types';
const COLORS = { null: '#e74c3c', kf: '#3498db' };

export function ModelComparison({ data }: { data: DashboardData }) {
  const summary = data.summary;

  const metrics = useMemo(() => {
    const avgKf = summary.reduce((s, r) => s + r.kf_rmse, 0) / summary.length;
    const avgNull = summary.reduce((s, r) => s + r.null_rmse, 0) / summary.length;
    const improvement = ((avgKf - avgNull) / avgKf) * 100;
    const total = summary.reduce((s, r) => s + (r.kf_count || 0), 0);
    return { avgKf, avgNull, improvement, total };
  }, [summary]);

  const rmseData = [
    {
      x: summary.map((r) => r.lead_time_hours),
      y: summary.map((r) => r.kf_rmse),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Kalman Filter',
      line: { color: COLORS.kf, width: 3 },
      marker: { size: 10 },
    },
    {
      x: summary.map((r) => r.lead_time_hours),
      y: summary.map((r) => r.null_rmse),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Null Model',
      line: { color: COLORS.null, width: 3 },
      marker: { size: 10 },
    },
  ];

  const meanData = [
    {
      x: summary.map((r) => r.lead_time_hours),
      y: summary.map((r) => r.kf_mean_error),
      type: 'bar',
      name: 'Kalman Filter',
      marker: { color: COLORS.kf },
    },
    {
      x: summary.map((r) => r.lead_time_hours),
      y: summary.map((r) => r.null_mean_error),
      type: 'bar',
      name: 'Null Model',
      marker: { color: COLORS.null },
    },
  ];

  const improvementPct = summary.map((r) =>
    ((r.kf_rmse - r.null_rmse) / r.kf_rmse) * 100
  );

  const improveData = [
    {
      x: summary.map((r) => r.lead_time_hours),
      y: improvementPct,
      type: 'bar',
      name: 'Improvement',
      marker: { color: COLORS.null },
      text: improvementPct.map((v) => `${v.toFixed(1)}%`),
      textposition: 'outside',
    },
  ];

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-800 to-indigo-600 bg-clip-text text-transparent">
          Model Comparison Overview
        </h1>
        <p className="text-slate-600 mt-2">
          High-level performance comparison: Null Model vs Kalman Filter
        </p>
      </header>

      <div className="bg-gradient-to-r from-pink-500 to-rose-500 rounded-xl p-6 text-white shadow-lg">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-8 h-8 shrink-0 mt-0.5" />
          <div>
            <h2 className="font-bold text-lg">Key Finding</h2>
            <p className="mt-1 text-white/95">
              The <strong>Null Model (simple persistence)</strong> consistently
              outperforms the <strong>Kalman Filter</strong> at all lead times,
              demonstrating that simpler approaches can sometimes achieve better
              results than sophisticated state-space models.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          icon={Target}
          label="Kalman Filter Avg RMSE"
          value={`${metrics.avgKf.toFixed(1)} km`}
          className="border-l-4 border-kf"
        />
        <MetricCard
          icon={Target}
          label="Null Model Avg RMSE"
          value={`${metrics.avgNull.toFixed(1)} km`}
          className="border-l-4 border-null"
        />
        <MetricCard
          icon={TrendingUp}
          label="Null Model Improvement"
          value={`${metrics.improvement.toFixed(1)}%`}
          sub="better"
          className="border-l-4 border-emerald-500"
        />
        <MetricCard
          icon={Target}
          label="Total Forecast Instances"
          value={metrics.total.toLocaleString()}
        />
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold mb-4">RMSE Comparison by Lead Time</h2>
        <Chart data={rmseData} layout={{ title: '', height: 500 }} />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-4">Mean Error Comparison</h2>
          <Chart
            data={meanData}
            layout={{
              barmode: 'group',
              title: '',
              xaxis: { title: 'Lead Time (hours)' },
              yaxis: { title: 'Mean Error (km)' },
              height: 400,
            }}
          />
        </div>
        <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-4">
            Null Model Improvement Over Kalman Filter
          </h2>
          <Chart
            data={improveData}
            layout={{
              shapes: [
                {
                  type: 'line',
                  x0: summary[0]?.lead_time_hours,
                  x1: summary[summary.length - 1]?.lead_time_hours,
                  y0: 0,
                  y1: 0,
                  line: { dash: 'dash', color: 'gray' },
                },
              ],
              xaxis: { title: 'Lead Time (hours)' },
              yaxis: { title: 'Improvement (%)' },
              height: 400,
            }}
          />
        </div>
      </div>

      <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-200 overflow-x-auto">
        <h2 className="text-xl font-semibold mb-4">Detailed Performance Metrics</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200">
              <th className="text-left py-3 px-4">Lead Time (h)</th>
              <th className="text-right py-3 px-4">KF Mean Error (km)</th>
              <th className="text-right py-3 px-4">KF RMSE (km)</th>
              <th className="text-right py-3 px-4">Null Mean Error (km)</th>
              <th className="text-right py-3 px-4">Null RMSE (km)</th>
              <th className="text-right py-3 px-4">Improvement (%)</th>
            </tr>
          </thead>
          <tbody>
            {summary.map((r) => (
              <tr key={r.lead_time_hours} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="py-3 px-4">{r.lead_time_hours}</td>
                <td className="text-right py-3 px-4">{r.kf_mean_error.toFixed(2)}</td>
                <td className="text-right py-3 px-4">{r.kf_rmse.toFixed(2)}</td>
                <td className="text-right py-3 px-4">{r.null_mean_error.toFixed(2)}</td>
                <td className="text-right py-3 px-4">{r.null_rmse.toFixed(2)}</td>
                <td className="text-right py-3 px-4 font-medium">
                  {(((r.kf_rmse - r.null_rmse) / r.kf_rmse) * 100).toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  sub,
  className = '',
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  sub?: string;
  className?: string;
}) {
  return (
    <div className={`bg-white rounded-xl p-4 shadow-sm border border-slate-200 ${className}`}>
      <div className="flex items-center gap-2 text-slate-600 text-sm mb-1">
        <Icon className="w-4 h-4" />
        {label}
      </div>
      <p className="text-2xl font-bold text-slate-800">{value}</p>
      {sub && <p className="text-sm text-emerald-600 font-medium">{sub}</p>}
    </div>
  );
}
