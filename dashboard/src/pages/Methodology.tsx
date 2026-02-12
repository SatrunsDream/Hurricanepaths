import { useState } from 'react';
import { FileText, Database, Cpu, BarChart3, Target, FileDown, Download } from 'lucide-react';

const REPORT_PDF = '/report.pdf';

const figures = [
  'fig3_storm_density.png',
  'fig4_curvature_histogram.png',
  'fig5_land_interaction.png',
  'fig7_kf_cycle.png',
  'fig8_forecast_error.png',
  'fig8a_forecast_error_rmse.png',
  'fig8b_error_distribution_by_leadtime.png',
  'fig9_error_distribution.png',
  'fig9a_rmse_distribution.png',
  'fig10_innovation_analysis.png',
  'fig11_model_comparison.png',
  'fig11a_rmse_comparison.png',
  'fig11b_error_distribution_comparison.png',
  'fig15a_error_spaghetti.png',
  'fig16_trajectory_spaghetti_24h.png',
];

const captions: Record<string, string> = {
  'fig3_storm_density.png': 'Figure 3: Storm Density Distribution',
  'fig4_curvature_histogram.png': 'Figure 4: Track Curvature Distribution',
  'fig5_land_interaction.png': 'Figure 5: Land Interaction Analysis',
  'fig7_kf_cycle.png': 'Figure 7: Kalman Filter Prediction-Update Cycle',
  'fig8_forecast_error.png': 'Figure 8: Forecast Error Analysis',
  'fig8a_forecast_error_rmse.png': 'Figure 8a: RMSE by Lead Time',
  'fig8b_error_distribution_by_leadtime.png': 'Figure 8b: Error Distribution by Lead Time',
  'fig9_error_distribution.png': 'Figure 9: Error Distribution Analysis',
  'fig9a_rmse_distribution.png': 'Figure 9a: RMSE Distribution',
  'fig10_innovation_analysis.png': 'Figure 10: Innovation Analysis',
  'fig11_model_comparison.png': 'Figure 11: Model Comparison Overview',
  'fig11a_rmse_comparison.png': 'Figure 11a: RMSE Comparison',
  'fig11b_error_distribution_comparison.png': 'Figure 11b: Error Distribution Comparison',
  'fig15a_error_spaghetti.png': 'Figure 15a: Error Trajectories (Spaghetti Plot)',
  'fig16_trajectory_spaghetti_24h.png': 'Figure 16: Trajectory Spaghetti Plot (24h)',
};

export function Methodology() {
  const [showReport, setShowReport] = useState(false);

  return (
    <div className="space-y-8 max-w-4xl">
      <header>
        <h1 className="text-3xl font-bold bg-gradient-to-r from-slate-800 to-indigo-600 bg-clip-text text-transparent">
          Methodology & Documentation
        </h1>
        <p className="text-slate-600 mt-2">
          Project overview, model implementations, and evaluation framework
        </p>
        <div className="flex gap-3 mt-4">
          <button
            onClick={() => setShowReport((v) => !v)}
            className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors font-medium"
          >
            <FileDown className="w-4 h-4" />
            {showReport ? 'Hide Report' : 'See Report'}
          </button>
          <a
            href={REPORT_PDF}
            download="Sequential_Bayesian_Inference_Applied_to_Hurricane_Tracking.pdf"
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-200 text-slate-700 rounded-lg hover:bg-slate-300 transition-colors font-medium"
          >
            <Download className="w-4 h-4" />
            Download PDF
          </a>
        </div>
      </header>

      {showReport && (
        <section className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
          <h2 className="text-xl font-semibold mb-4">Sequential Bayesian Inference Applied to Hurricane Tracking</h2>
          <iframe
            src={`${REPORT_PDF}#toolbar=1`}
            title="Project Report PDF"
            className="w-full h-[calc(100vh-12rem)] min-h-[600px] rounded-lg border border-slate-200"
          />
        </section>
      )}

      <section className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold flex items-center gap-2 mb-4">
          <FileText className="w-5 h-5" />
          Project Overview
        </h2>
        <p className="text-slate-600 leading-relaxed">
          This project implements and evaluates probabilistic state-space models for hurricane
          track forecasting using sequential Bayesian inference. The primary model is a{' '}
          <strong>Kalman Filter</strong> with adaptive process noise, compared against a{' '}
          <strong>Null Model</strong> baseline using simple velocity persistence. The evaluation
          reveals a surprising finding: the simple persistence baseline consistently outperforms
          the sophisticated Kalman Filter at all lead times.
        </p>
        <ul className="mt-4 space-y-2 text-slate-600">
          <li><strong>Dataset:</strong> IBTrACS (International Best Track Archive)</li>
          <li><strong>Temporal Range:</strong> 1842–2025 (183 years)</li>
          <li><strong>Total Observations:</strong> 721,960 across 13,450 storms</li>
        </ul>
      </section>

      <section className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold flex items-center gap-2 mb-4">
          <Database className="w-5 h-5" />
          Data Processing
        </h2>
        <p className="text-slate-600 leading-relaxed">
          The IBTrACS dataset contains hurricane track data with specialized parsing for two-header
          format, column normalization, and temporal structure validation. Velocity is computed
          from position differences using haversine distance. The state vector is [lat, lon, v_lat,
          v_lon] converted to metric coordinates for the Kalman filter.
        </p>
      </section>

      <section className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold flex items-center gap-2 mb-4">
          <Cpu className="w-5 h-5" />
          Model Implementations
        </h2>
        <p className="text-slate-600 leading-relaxed">
          The Kalman filter uses a <strong>constant velocity model</strong> with state vector
          [x_km, y_km, vx_km, vy_km]. Process noise is estimated from training data; observation
          noise reflects best-track uncertainty. The null model implements persistence: velocity
          from the last time step is used to extrapolate position forward.
        </p>
      </section>

      <section className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold flex items-center gap-2 mb-4">
          <BarChart3 className="w-5 h-5" />
          Key Finding
        </h2>
        <div className="bg-rose-50 border border-rose-200 rounded-lg p-4">
          <p className="text-rose-900 font-medium">
            The Null Model (simple persistence) consistently outperforms the Kalman Filter at all
            lead times. Possible explanations include suboptimal parameter tuning, error
            accumulation through recursive updating, or that direct velocity persistence is more
            effective than filtered state estimates.
          </p>
        </div>
      </section>

      <section className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold flex items-center gap-2 mb-4">
          <Target className="w-5 h-5" />
          Figures
        </h2>
        <p className="text-slate-600 mb-6">
          Copy the <code className="bg-slate-200 px-1 rounded">figures/</code> folder from the
          project root into <code className="bg-slate-200 px-1 rounded">dashboard/public/figures/</code> to
          display analysis figures below.
        </p>
        <div className="grid gap-6">
          {figures.map((fig) => (
            <figure key={fig} className="border border-slate-200 rounded-lg overflow-hidden">
              <img
                src={`/figures/${fig}`}
                alt={captions[fig] || fig}
                className="w-full"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
              <figcaption className="p-3 bg-slate-50 text-sm text-slate-600">
                {captions[fig] || fig}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
        <h2 className="text-xl font-semibold mb-4">References</h2>
        <ul className="text-slate-600 space-y-2 text-sm">
          <li>
            IBTrACS Documentation:{' '}
            <a href="https://www.ncei.noaa.gov/products/international-best-track-archive" className="text-indigo-600 hover:underline">
              NCEI
            </a>
          </li>
          <li>Bril, G. (1995): Forecasting hurricane tracks using the Kalman filter. Environmetrics.</li>
          <li>Sequential Bayesian Inference Applied to Hurricane Tracking</li>
        </ul>
      </section>
    </div>
  );
}
