export interface SummaryRow {
  lead_time_hours: number;
  kf_mean_error: number;
  kf_rmse: number;
  kf_count: number;
  null_mean_error: number;
  null_rmse: number;
  null_count: number;
  rmse_improvement_pct?: number;
}

export interface ErrorRow {
  sid: string;
  origin_idx: number;
  lead_time_hours: number;
  error_km: number;
  model: string;
  basin?: string;
  nature?: string;
  error_category?: string;
}

export interface StormRow {
  sid: string;
  kf_mean_error: number;
  null_mean_error: number;
  basin?: string;
  nature?: string;
  better_model: string;
}

export interface ComparisonRow {
  sid: string;
  lead_time_hours: number;
  kf_error_km: number;
  null_error_km: number;
  error_difference?: number;
  null_better?: boolean;
}

export interface MetadataRow {
  sid: string;
  basin?: string;
  basin_name?: string;
  nature?: string;
  storm_type?: string;
}

export interface DashboardData {
  summary: SummaryRow[];
  errors: ErrorRow[];
  storms: StormRow[];
  comparison: ComparisonRow[];
  metadata: MetadataRow[];
}
