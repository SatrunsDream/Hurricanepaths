"""
Comprehensive Data Preprocessing for Hurricane Model Comparison Dashboard

This script creates multiple optimized CSV files for visualization:
1. model_comparison_summary.csv - High-level performance metrics by lead time
2. error_distributions.csv - Detailed error data for distribution plots
3. storm_performance.csv - Per-storm performance metrics
4. model_comparison_detailed.csv - Side-by-side error comparisons
5. storm_metadata.csv - Storm characteristics and metadata
"""

import pandas as pd
import numpy as np
import os
import pickle

def get_data_dir():
    """Get the data directory path"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_repo_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(main_repo_dir, "data")
    return data_dir, main_repo_dir

def load_track_data():
    """Load hurricane track data with metadata"""
    data_dir, main_repo_dir = get_data_dir()
    
    possible_files = [
        os.path.join(data_dir, "hurricane_paths_processed.pkl"),
        os.path.join(main_repo_dir, "hurricane_paths_processed.pkl"),
        os.path.join(data_dir, "hurricane_paths_processed_MODEL.pkl"),
        os.path.join(main_repo_dir, "hurricane_paths_processed_MODEL.pkl"),
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            print(f"Loading track data from: {filepath}")
            return pd.read_pickle(filepath)
    
    raise FileNotFoundError(f"Could not find track data file. Checked: {possible_files}")

def load_kalman_results():
    """Load Kalman filter forecast results"""
    data_dir, main_repo_dir = get_data_dir()
    
    test_files = [
        os.path.join(data_dir, "sliding_results_final.pkl"),
        os.path.join(data_dir, "kalman_results_sliding_results_final.pkl"),
        os.path.join(main_repo_dir, "sliding_results_final.pkl"),
    ]
    
    val_files = [
        os.path.join(data_dir, "val_sliding_results_final.pkl"),
        os.path.join(data_dir, "kalman_results_val_sliding_results_final.pkl"),
        os.path.join(main_repo_dir, "val_sliding_results_final.pkl"),
    ]
    
    test_results = None
    for filepath in test_files:
        if os.path.exists(filepath):
            print(f"Loading KF test results from: {filepath}")
            test_results = pd.read_pickle(filepath)
            break
    
    val_results = None
    for filepath in val_files:
        if os.path.exists(filepath):
            print(f"Loading KF validation results from: {filepath}")
            val_results = pd.read_pickle(filepath)
            break
    
    if test_results is not None and val_results is not None:
        test_results = test_results.copy()
        test_results['dataset'] = 'test'
        val_results = val_results.copy()
        val_results['dataset'] = 'validation'
        combined = pd.concat([test_results, val_results], ignore_index=True)
        print(f"Combined KF results: {len(combined)} forecast instances")
        return combined
    elif test_results is not None:
        print(f"Using KF test results only: {len(test_results)} forecast instances")
        return test_results
    elif val_results is not None:
        print(f"Using KF validation results only: {len(val_results)} forecast instances")
        return val_results
    else:
        raise FileNotFoundError("Could not find Kalman filter results")

def load_null_model_results():
    """Load null model forecast results"""
    data_dir, main_repo_dir = get_data_dir()
    
    test_file = os.path.join(data_dir, "null_model_test_results.pkl")
    val_file = os.path.join(data_dir, "null_model_val_results.pkl")
    
    test_results = None
    val_results = None
    
    if os.path.exists(test_file):
        print(f"Loading null model test results from: {test_file}")
        test_results = pd.read_pickle(test_file)
    
    if os.path.exists(val_file):
        print(f"Loading null model validation results from: {val_file}")
        val_results = pd.read_pickle(val_file)
    
    if test_results is not None and val_results is not None:
        test_results = test_results.copy()
        test_results['dataset'] = 'test'
        val_results = val_results.copy()
        val_results['dataset'] = 'validation'
        combined = pd.concat([test_results, val_results], ignore_index=True)
        print(f"Combined null model results: {len(combined)} forecast instances")
        return combined
    elif test_results is not None:
        print(f"Using null model test results only: {len(test_results)} forecast instances")
        return test_results
    elif val_results is not None:
        print(f"Using null model validation results only: {len(val_results)} forecast instances")
        return val_results
    else:
        raise FileNotFoundError("Could not find null model results")

def standardize_storm_id(df):
    """Standardize storm ID column name"""
    if 'storm_id' in df.columns:
        df = df.rename(columns={'storm_id': 'sid'})
    return df

def create_model_comparison_summary(kf_results, null_results):
    """Create summary CSV with performance metrics by lead time"""
    lead_times = sorted(kf_results['lead_time_hours'].unique())
    
    summary_data = []
    for lt in lead_times:
        kf_lt = kf_results[kf_results['lead_time_hours'] == lt]['error_km']
        null_lt = null_results[null_results['lead_time_hours'] == lt]['error_km']
        
        summary_data.append({
            'lead_time_hours': lt,
            'kf_mean_error': kf_lt.mean(),
            'kf_median_error': kf_lt.median(),
            'kf_rmse': np.sqrt(np.mean(kf_lt**2)),
            'kf_std_error': kf_lt.std(),
            'kf_count': len(kf_lt),
            'null_mean_error': null_lt.mean(),
            'null_median_error': null_lt.median(),
            'null_rmse': np.sqrt(np.mean(null_lt**2)),
            'null_std_error': null_lt.std(),
            'null_count': len(null_lt),
            'rmse_improvement_pct': ((null_lt.mean() - kf_lt.mean()) / null_lt.mean() * 100) if null_lt.mean() > 0 else 0,
            'null_better': np.sqrt(np.mean(null_lt**2)) < np.sqrt(np.mean(kf_lt**2))
        })
    
    return pd.DataFrame(summary_data)

def create_error_distributions(kf_results, null_results, track_data):
    """Create detailed error distribution data for plotting"""
    kf_results = kf_results.copy()
    kf_results['model'] = 'Kalman Filter'
    
    null_results = null_results.copy()
    null_results['model'] = 'Null Model'
    
    combined = pd.concat([kf_results, null_results], ignore_index=True)
    
    # Add storm metadata
    storm_meta = track_data.groupby('sid').agg({
        'basin': 'first',
        'nature': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
    }).reset_index()
    
    # Merge with metadata
    combined = combined.merge(storm_meta, on='sid', how='left')
    
    # Add derived columns
    combined['error_squared'] = combined['error_km'] ** 2
    combined['error_category'] = pd.cut(
        combined['error_km'],
        bins=[0, 25, 50, 100, 200, np.inf],
        labels=['Excellent (<25km)', 'Good (25-50km)', 'Fair (50-100km)', 'Poor (100-200km)', 'Very Poor (>200km)']
    )
    
    return combined

def create_storm_performance(kf_results, null_results, track_data):
    """Create per-storm performance metrics"""
    # Aggregate by storm
    kf_storm = kf_results.groupby('sid').agg({
        'error_km': ['mean', 'std', 'count'],
        'lead_time_hours': 'first'
    }).reset_index()
    kf_storm.columns = ['sid', 'kf_mean_error', 'kf_std_error', 'kf_forecast_count', 'lead_time_hours']
    
    null_storm = null_results.groupby('sid').agg({
        'error_km': ['mean', 'std', 'count']
    }).reset_index()
    null_storm.columns = ['sid', 'null_mean_error', 'null_std_error', 'null_forecast_count']
    
    # Merge
    storm_perf = kf_storm.merge(null_storm, on='sid', how='outer')
    
    # Add storm metadata
    storm_meta = track_data.groupby('sid').agg({
        'basin': 'first',
        'nature': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'iso_time': ['min', 'max'],
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    
    # Flatten MultiIndex if present
    if isinstance(storm_meta.columns, pd.MultiIndex):
        new_cols = []
        for col in storm_meta.columns:
            if isinstance(col, tuple):
                if col[0] == 'iso_time':
                    if col[1] == 'min':
                        new_cols.append('start_time')
                    elif col[1] == 'max':
                        new_cols.append('end_time')
                    else:
                        new_cols.append(f'{col[0]}_{col[1]}')
                else:
                    new_cols.append(col[0])
            else:
                new_cols.append(str(col))
        storm_meta.columns = new_cols
    
    storm_perf = storm_perf.merge(storm_meta, on='sid', how='left')
    
    # Calculate which model performs better
    storm_perf['better_model'] = 'Kalman Filter'
    mask = storm_perf['null_mean_error'] < storm_perf['kf_mean_error']
    storm_perf.loc[mask, 'better_model'] = 'Null Model'
    
    return storm_perf

def create_model_comparison_detailed(kf_results, null_results):
    """Create side-by-side comparison for each forecast instance"""
    # Merge on storm_id, origin_idx, and lead_time_hours
    kf_merge = kf_results[['sid', 'origin_idx', 'lead_time_hours', 'error_km']].copy()
    kf_merge.columns = ['sid', 'origin_idx', 'lead_time_hours', 'kf_error_km']
    
    null_merge = null_results[['sid', 'origin_idx', 'lead_time_hours', 'error_km']].copy()
    null_merge.columns = ['sid', 'origin_idx', 'lead_time_hours', 'null_error_km']
    
    comparison = kf_merge.merge(
        null_merge,
        on=['sid', 'origin_idx', 'lead_time_hours'],
        how='inner'
    )
    
    comparison['error_difference'] = comparison['null_error_km'] - comparison['kf_error_km']
    comparison['null_better'] = comparison['error_difference'] < 0
    comparison['improvement_km'] = np.abs(comparison['error_difference'])
    
    return comparison

def create_storm_metadata(track_data):
    """Create storm metadata CSV"""
    # Build aggregation dict based on available columns
    agg_dict = {
        'iso_time': ['min', 'max', 'count'],
        'lat': ['first', 'min', 'max'],
        'lon': ['first', 'min', 'max'],
    }
    
    # Add optional columns if they exist
    if 'basin' in track_data.columns:
        agg_dict['basin'] = 'first'
    if 'nature' in track_data.columns:
        agg_dict['nature'] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    if 'storm_speed' in track_data.columns:
        agg_dict['storm_speed'] = 'mean'
    if 'storm_dir' in track_data.columns:
        agg_dict['storm_dir'] = 'mean'
    
    storm_meta = track_data.groupby('sid').agg(agg_dict).reset_index()
    
    # Flatten MultiIndex
    if isinstance(storm_meta.columns, pd.MultiIndex):
        new_cols = ['sid']
        for col in storm_meta.columns[1:]:
            if isinstance(col, tuple):
                col_name, agg_func = col[0], col[1]
                if col_name == 'iso_time':
                    if agg_func == 'min':
                        new_cols.append('start_time')
                    elif agg_func == 'max':
                        new_cols.append('end_time')
                    elif agg_func == 'count':
                        new_cols.append('observation_count')
                    else:
                        new_cols.append(f'{col_name}_{agg_func}')
                elif col_name == 'lat':
                    if agg_func == 'first':
                        new_cols.append('lat_first')
                    elif agg_func == 'min':
                        new_cols.append('lat_min')
                    elif agg_func == 'max':
                        new_cols.append('lat_max')
                    else:
                        new_cols.append(f'{col_name}_{agg_func}')
                elif col_name == 'lon':
                    if agg_func == 'first':
                        new_cols.append('lon_first')
                    elif agg_func == 'min':
                        new_cols.append('lon_min')
                    elif agg_func == 'max':
                        new_cols.append('lon_max')
                    else:
                        new_cols.append(f'{col_name}_{agg_func}')
                elif col_name in ['basin', 'nature']:
                    # These should just be 'basin' or 'nature' since they use 'first'
                    new_cols.append(col_name)
                else:
                    new_cols.append(f'{col_name}_{agg_func}')
            else:
                new_cols.append(str(col))
        storm_meta.columns = new_cols
    
    # Extract season from start_time
    if 'start_time' in storm_meta.columns:
        storm_meta['season'] = pd.to_datetime(storm_meta['start_time']).dt.year
        storm_meta['duration_days'] = (
            (pd.to_datetime(storm_meta['end_time']) - pd.to_datetime(storm_meta['start_time'])).dt.total_seconds() / 86400
        )
    
    # Add basin name mapping if basin column exists
    if 'basin' in storm_meta.columns:
        basin_mapping = {
            'NA': 'North Atlantic',
            'EP': 'Eastern Pacific',
            'WP': 'Western Pacific',
            'NI': 'North Indian',
            'SI': 'South Indian',
            'SP': 'South Pacific',
            'SA': 'South Atlantic'
        }
        storm_meta['basin_name'] = storm_meta['basin'].map(basin_mapping).fillna(storm_meta['basin'])
    else:
        storm_meta['basin'] = None
        storm_meta['basin_name'] = None
    
    # Add storm type mapping if nature column exists
    if 'nature' in storm_meta.columns:
        type_mapping = {
            'TS': 'Tropical Storm',
            'ET': 'Extratropical',
            'DS': 'Disturbance',
            'MX': 'Mixed',
            'SS': 'Subtropical Storm',
            'TD': 'Tropical Depression',
            'HU': 'Hurricane'
        }
        storm_meta['storm_type'] = storm_meta['nature'].map(type_mapping).fillna(storm_meta['nature'])
    else:
        storm_meta['nature'] = None
        storm_meta['storm_type'] = None
    
    return storm_meta

def main():
    """Main preprocessing function"""
    print("=" * 70)
    print("STREAMBOARD DATA PREPROCESSING")
    print("=" * 70)
    
    # Get output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    
    # Load data
    print("\n[1/5] Loading track data...")
    track_data = load_track_data()
    print(f"   Loaded {len(track_data):,} observations from {track_data['sid'].nunique():,} storms")
    
    print("\n[2/5] Loading Kalman filter results...")
    kf_results = load_kalman_results()
    kf_results = standardize_storm_id(kf_results)
    print(f"   Loaded {len(kf_results):,} KF forecast instances")
    
    print("\n[3/5] Loading null model results...")
    null_results = load_null_model_results()
    null_results = standardize_storm_id(null_results)
    print(f"   Loaded {len(null_results):,} null model forecast instances")
    
    # Create CSV files
    print("\n[4/5] Creating optimized CSV files...")
    
    print("   Creating model_comparison_summary.csv...")
    summary_df = create_model_comparison_summary(kf_results, null_results)
    summary_df.to_csv(os.path.join(output_dir, "model_comparison_summary.csv"), index=False)
    print(f"      Saved {len(summary_df)} rows")
    
    print("   Creating error_distributions.csv...")
    error_dist_df = create_error_distributions(kf_results, null_results, track_data)
    error_dist_df.to_csv(os.path.join(output_dir, "error_distributions.csv"), index=False)
    print(f"      Saved {len(error_dist_df):,} rows")
    
    print("   Creating storm_performance.csv...")
    storm_perf_df = create_storm_performance(kf_results, null_results, track_data)
    storm_perf_df.to_csv(os.path.join(output_dir, "storm_performance.csv"), index=False)
    print(f"      Saved {len(storm_perf_df):,} rows")
    
    print("   Creating model_comparison_detailed.csv...")
    comparison_df = create_model_comparison_detailed(kf_results, null_results)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison_detailed.csv"), index=False)
    print(f"      Saved {len(comparison_df):,} rows")
    
    print("   Creating storm_metadata.csv...")
    metadata_df = create_storm_metadata(track_data)
    metadata_df.to_csv(os.path.join(output_dir, "storm_metadata.csv"), index=False)
    print(f"      Saved {len(metadata_df):,} rows")
    
    # Print summary
    print("\n[5/5] Summary Statistics")
    print("=" * 70)
    print("\nModel Comparison Summary:")
    print(summary_df.to_string(index=False))
    
    print(f"\n\nAll CSV files saved to: {output_dir}")
    print("\nFiles created:")
    print("  - model_comparison_summary.csv")
    print("  - error_distributions.csv")
    print("  - storm_performance.csv")
    print("  - model_comparison_detailed.csv")
    print("  - storm_metadata.csv")
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
