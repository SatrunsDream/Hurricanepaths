import Papa from 'papaparse';
import type { DashboardData, SummaryRow, ErrorRow, StormRow, ComparisonRow, MetadataRow } from '../types';

const DATA_PATH = '/data';

async function fetchCSV<T>(filename: string): Promise<T[]> {
  const res = await fetch(`${DATA_PATH}/${filename}`);
  if (!res.ok) throw new Error(`Failed to load ${filename}`);
  const text = await res.text();
  const { data } = Papa.parse<Record<string, string>>(text, { header: true, skipEmptyLines: true });
  return data.map((row) => {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(row)) {
      const num = parseFloat(v);
      out[k] = !isNaN(num) && v !== '' ? num : v;
    }
    return out as T;
  });
}

export async function loadData(): Promise<DashboardData> {
  const [summary, errors, storms, comparison, metadata] = await Promise.all([
    fetchCSV<SummaryRow>('model_comparison_summary.csv'),
    fetchCSV<ErrorRow>('error_distributions.csv'),
    fetchCSV<StormRow>('storm_performance.csv'),
    fetchCSV<ComparisonRow>('model_comparison_detailed.csv'),
    fetchCSV<MetadataRow>('storm_metadata.csv'),
  ]);

  return { summary, errors, storms, comparison, metadata };
}
