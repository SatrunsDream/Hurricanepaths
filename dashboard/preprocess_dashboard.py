"""
Data preprocessing script for Hurricane Tracking Dashboard

This script merges hurricane track metadata with forecast error results
to create a lightweight CSV file for the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import os

def load_track_data():
    """Load hurricane track data with metadata"""
    # Get the directory where this script is located, then go up to main repo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_repo_dir = os.path.dirname(script_dir)  # Go up one level from dashboard/ to main repo
    
    # Try different possible file names - prioritize the full dataset that matches forecast results
    possible_files = [
        os.path.join(main_repo_dir, "data/hurricane_paths_processed.pkl"),  # Full dataset (1842-2025) - matches forecast results
        os.path.join(main_repo_dir, "hurricane_paths_processed.pkl"),
        os.path.join(main_repo_dir, "data/hurricane_paths_processed_MODEL.pkl"),  # Filtered dataset (1945-2024) - may not match
        os.path.join(main_repo_dir, "hurricane_paths_processed_MODEL.pkl"),
        "data/hurricane_paths_processed.pkl",  # Fallback: relative paths
        "hurricane_paths_processed.pkl",
        "data/hurricane_paths_processed_MODEL.pkl",
        "hurricane_paths_processed_MODEL.pkl"
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            print(f"Loading track data from: {filepath}")
            df = pd.read_pickle(filepath)
            return df
    
    raise FileNotFoundError(
        "Could not find track data file. Please ensure one of these exists:\n" +
        "\n".join(possible_files[:4])  # Show main repo paths
    )

def load_forecast_results():
    """Load forecast error results (test and validation sets)"""
    # Get the directory where this script is located, then go up to main repo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_repo_dir = os.path.dirname(script_dir)  # Go up one level from dashboard/ to main repo
    
    # Try to load test set results
    test_files = [
        os.path.join(main_repo_dir, "data/sliding_results_final.pkl"),  # Primary path
        os.path.join(main_repo_dir, "data/kalman_results_sliding_results_final.pkl"),
        os.path.join(main_repo_dir, "sliding_results_final.pkl"),
        os.path.join(main_repo_dir, "kalman_results_sliding_results_final.pkl"),
        "data/sliding_results_final.pkl",  # Fallback: relative paths
        "data/kalman_results_sliding_results_final.pkl",
        "sliding_results_final.pkl",
        "kalman_results_sliding_results_final.pkl"
    ]
    
    test_results = None
    for filepath in test_files:
        if os.path.exists(filepath):
            print(f"Loading test set results from: {filepath}")
            test_results = pd.read_pickle(filepath)
            break
    
    # Try to load validation set results
    val_files = [
        os.path.join(main_repo_dir, "data/val_sliding_results_final.pkl"),  # Primary path
        os.path.join(main_repo_dir, "data/kalman_results_val_sliding_results_final.pkl"),
        os.path.join(main_repo_dir, "val_sliding_results_final.pkl"),
        os.path.join(main_repo_dir, "kalman_results_val_sliding_results_final.pkl"),
        "data/val_sliding_results_final.pkl",  # Fallback: relative paths
        "data/kalman_results_val_sliding_results_final.pkl",
        "val_sliding_results_final.pkl",
        "kalman_results_val_sliding_results_final.pkl"
    ]
    
    val_results = None
    for filepath in val_files:
        if os.path.exists(filepath):
            print(f"Loading validation set results from: {filepath}")
            val_results = pd.read_pickle(filepath)
            break
    
    # Combine test and validation results if both exist
    if test_results is not None and val_results is not None:
        print(f"Combining test ({len(test_results)} rows) and validation ({len(val_results)} rows) results")
        # Add a dataset indicator
        test_results = test_results.copy()
        test_results['dataset'] = 'test'
        val_results = val_results.copy()
        val_results['dataset'] = 'validation'
        combined = pd.concat([test_results, val_results], ignore_index=True)
        print(f"Combined dataset: {len(combined)} forecast instances")
        return combined
    elif test_results is not None:
        print(f"Using test set results only: {len(test_results)} forecast instances")
        return test_results
    elif val_results is not None:
        print(f"Using validation set results only: {len(val_results)} forecast instances")
        return val_results
    else:
        raise FileNotFoundError(
            "Could not find forecast results file. Please ensure one of these exists:\n" +
            "\nTest set:\n" + "\n".join(test_files) +
            "\nValidation set:\n" + "\n".join(val_files) +
            "\n\nIf you haven't generated forecast results yet, you'll need to run the Kalman Filter evaluation first."
        )

def preprocess_dashboard_data():
    """Merge track data and forecast results into dashboard-ready format"""
    
    print("=" * 60)
    print("HURRICANE TRACKING DASHBOARD - DATA PREPROCESSING")
    print("=" * 60)
    
    # Load track data
    print("\n[1/4] Loading track data...")
    track_df = load_track_data()
    print(f"   Loaded {len(track_df):,} observations from {track_df['sid'].nunique():,} storms")
    
    # INSPECT TRACK DATA COLUMNS
    print("\n" + "=" * 60)
    print("TRACK DATA COLUMNS AND DATA TYPES:")
    print("=" * 60)
    print(f"Total columns: {len(track_df.columns)}")
    print("\nColumn names and data types:")
    for i, (col, dtype) in enumerate(track_df.dtypes.items(), 1):
        null_count = track_df[col].isna().sum()
        null_pct = (null_count / len(track_df)) * 100
        print(f"  {i:3d}. {col:30s} | {str(dtype):15s} | Nulls: {null_count:6,} ({null_pct:5.1f}%)")
    
    # Check for key columns
    print("\n" + "=" * 60)
    print("CHECKING FOR KEY COLUMNS:")
    print("=" * 60)
    key_columns = ['sid', 'basin', 'nature', 'iso_time', 'lat', 'lon', 'season']
    available_cols = {}
    for col in key_columns:
        exists = col in track_df.columns
        available_cols[col] = exists
        status = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {col:15s}: {status}")
    
    # Extract unique storm metadata (one row per storm)
    print("\n[2/4] Extracting storm metadata...")
    
    # Build aggregation dictionary based on available columns
    agg_dict = {
        'basin': 'first',
        'nature': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
        'iso_time': ['min', 'max'],
        'lat': 'first',
        'lon': 'first',
    }
    
    # Add season if available, otherwise extract from iso_time
    if 'season' in track_df.columns:
        agg_dict['season'] = 'first'
    
    storm_metadata = track_df.groupby('sid').agg(agg_dict).reset_index()
    
    # Flatten column names (handle multi-level columns from iso_time)
    # First, check if we have MultiIndex columns
    if isinstance(storm_metadata.columns, pd.MultiIndex):
        # Flatten MultiIndex columns
        new_cols = []
        for col in storm_metadata.columns:
            if isinstance(col, tuple):
                if len(col) == 2:
                    # Handle ('column_name', 'agg_func') format
                    col_name, agg_func = col
                    if col_name == 'iso_time':
                        if agg_func == 'min':
                            new_cols.append('start_time')
                        elif agg_func == 'max':
                            new_cols.append('end_time')
                        else:
                            new_cols.append(f'{col_name}_{agg_func}')
                    else:
                        new_cols.append(col_name)
                else:
                    # Fallback: use first element
                    new_cols.append(str(col[0]) if len(col) > 0 else str(col))
            else:
                new_cols.append(str(col))
        
        # Create a new DataFrame with flattened columns to ensure single-level index
        storm_metadata = pd.DataFrame(storm_metadata.values, columns=new_cols, index=storm_metadata.index)
    
    # Now ensure column names match expected format
    expected_cols = ['sid', 'basin', 'nature', 'start_time', 'end_time', 'start_lat', 'start_lon']
    if 'season' in agg_dict:
        expected_cols.append('season')
    
    # Create rename dictionary for any mismatches
    rename_dict = {}
    current_cols = list(storm_metadata.columns)
    
    # Map known patterns
    for i, curr_col in enumerate(current_cols):
        curr_col_str = str(curr_col)
        # Handle sid column
        if curr_col_str == 'sid' or i == 0:
            continue  # Already correct
        # Map known columns
        elif 'start_time' in curr_col_str.lower() or 'min' in curr_col_str.lower():
            if 'start_time' not in [c for c in rename_dict.values()]:
                rename_dict[curr_col] = 'start_time'
        elif 'end_time' in curr_col_str.lower() or 'max' in curr_col_str.lower():
            if 'end_time' not in [c for c in rename_dict.values()]:
                rename_dict[curr_col] = 'end_time'
        elif curr_col_str == 'basin':
            continue  # Already correct
        elif curr_col_str == 'nature':
            continue  # Already correct
        elif curr_col_str == 'lat' or 'lat' in curr_col_str.lower():
            if 'start_lat' not in [c for c in rename_dict.values()]:
                rename_dict[curr_col] = 'start_lat'
        elif curr_col_str == 'lon' or 'lon' in curr_col_str.lower():
            if 'start_lon' not in [c for c in rename_dict.values()]:
                rename_dict[curr_col] = 'start_lon'
        elif curr_col_str == 'season':
            continue  # Already correct
    
    if rename_dict:
        storm_metadata = storm_metadata.rename(columns=rename_dict)
    
    # Ensure we have the expected columns, fill missing with NaN
    for col in expected_cols:
        if col not in storm_metadata.columns:
            if col == 'season':
                storm_metadata[col] = None
            else:
                storm_metadata[col] = None
    
    # Extract season from iso_time if not available
    if 'season' not in storm_metadata.columns and 'start_time' in storm_metadata.columns:
        storm_metadata['season'] = pd.to_datetime(storm_metadata['start_time']).dt.year
    
    # Calculate storm duration in days
    if 'start_time' in storm_metadata.columns and 'end_time' in storm_metadata.columns:
        storm_metadata['duration_days'] = (
            (pd.to_datetime(storm_metadata['end_time']) - pd.to_datetime(storm_metadata['start_time'])).dt.total_seconds() / 86400
        )
    else:
        storm_metadata['duration_days'] = np.nan
    
    # Final check: ensure columns are single-level (not MultiIndex)
    if isinstance(storm_metadata.columns, pd.MultiIndex):
        # Force flatten by creating new DataFrame
        storm_metadata = pd.DataFrame(
            storm_metadata.values, 
            columns=[str(c) for c in storm_metadata.columns.get_level_values(0)],
            index=storm_metadata.index
        )
    
    print(f"   Extracted metadata for {len(storm_metadata):,} storms")
    print(f"   Metadata columns: {list(storm_metadata.columns)}")
    print(f"   Column index type: {type(storm_metadata.columns)}")
    
    # Load forecast results
    print("\n[3/4] Loading forecast results...")
    results_df = load_forecast_results()
    print(f"   Loaded {len(results_df):,} forecast instances")
    
    # INSPECT FORECAST RESULTS COLUMNS
    print("\n" + "=" * 60)
    print("FORECAST RESULTS COLUMNS AND DATA TYPES:")
    print("=" * 60)
    print(f"Total columns: {len(results_df.columns)}")
    print("\nColumn names and data types:")
    for i, (col, dtype) in enumerate(results_df.dtypes.items(), 1):
        null_count = results_df[col].isna().sum()
        null_pct = (null_count / len(results_df)) * 100 if len(results_df) > 0 else 0
        print(f"  {i:3d}. {col:30s} | {str(dtype):15s} | Nulls: {null_count:6,} ({null_pct:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("CHECKING FOR REQUIRED COLUMNS:")
    print("=" * 60)
    required_result_cols = ['lead_time_hours', 'error_km']
    for col in required_result_cols:
        exists = col in results_df.columns
        status = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  {col:20s}: {status}")
    
    # Check column names and standardize
    if 'storm_id' in results_df.columns:
        results_df = results_df.rename(columns={'storm_id': 'sid'})
    elif 'sid' not in results_df.columns:
        raise ValueError(
            "Forecast results must contain 'sid' or 'storm_id' column. "
            f"Found columns: {list(results_df.columns)}"
        )
    
    # Ensure required columns exist
    required_cols = ['lead_time_hours', 'error_km']
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        raise ValueError(
            f"Forecast results missing required columns: {missing_cols}. "
            f"Found columns: {list(results_df.columns)}"
        )
    
    print(f"   Forecast results contain lead times: {sorted(results_df['lead_time_hours'].unique())}")
    
    # Merge results with storm metadata
    print("\n[4/4] Merging data...")
    
    # Debug: Check sid values match
    results_sids = set(results_df['sid'].unique())
    metadata_sids = set(storm_metadata['sid'].unique())
    matching_sids = results_sids.intersection(metadata_sids)
    
    print(f"   Results file has {len(results_sids)} unique storm IDs")
    print(f"   Metadata has {len(metadata_sids)} unique storm IDs")
    print(f"   Matching storm IDs: {len(matching_sids)}")
    
    if len(matching_sids) == 0:
        print(f"   WARNING: No matching storm IDs found!")
        print(f"   Sample results SIDs: {list(results_sids)[:5]}")
        print(f"   Sample metadata SIDs: {list(metadata_sids)[:5]}")
    
    dashboard_data = results_df.merge(
        storm_metadata,
        on='sid',
        how='left'
    )
    
    # Check merge success
    merged_count = dashboard_data['basin'].notna().sum()
    print(f"   Successfully merged {merged_count:,} forecast instances with metadata")
    
    if merged_count < len(dashboard_data) * 0.9:
        print(f"   WARNING: Only {merged_count/len(dashboard_data)*100:.1f}% of forecasts matched with storm metadata")
    
    # Ensure we have at least some metadata - fill from first non-null if needed
    if merged_count > 0:
        # For storms with missing metadata, try to fill from other rows of same storm
        for sid in dashboard_data['sid'].unique():
            storm_rows = dashboard_data[dashboard_data['sid'] == sid]
            if storm_rows['basin'].isna().all():
                # This storm has no metadata - skip
                continue
            # Fill NaN values with first non-null value for this storm
            for col in ['basin', 'nature', 'start_time', 'end_time', 'start_lat', 'start_lon', 'season', 'duration_days']:
                if col in dashboard_data.columns:
                    non_null_val = storm_rows[col].dropna()
                    if len(non_null_val) > 0:
                        fill_value = non_null_val.iloc[0]
                        # Use explicit assignment to avoid FutureWarning about downcasting
                        mask = (dashboard_data['sid'] == sid) & (dashboard_data[col].isna())
                        dashboard_data.loc[mask, col] = fill_value
    
    # Add derived columns for easier filtering
    basin_mapping = dashboard_data['basin'].map({
        'NA': 'North Atlantic',
        'EP': 'Eastern Pacific',
        'WP': 'Western Pacific',
        'NI': 'North Indian',
        'SI': 'South Indian',
        'SP': 'South Pacific',
        'SA': 'South Atlantic'
    })
    dashboard_data['basin_name'] = basin_mapping.fillna(dashboard_data['basin']).astype(str)
    
    # Create storm type categories
    storm_type_mapping = dashboard_data['nature'].map({
        'TS': 'Tropical Storm',
        'ET': 'Extratropical',
        'DS': 'Disturbance',
        'MX': 'Mixed',
        'SS': 'Subtropical Storm',
        'TD': 'Tropical Depression',
        'HU': 'Hurricane'
    })
    dashboard_data['storm_type'] = storm_type_mapping.fillna(dashboard_data['nature']).astype(str)
    
    # Save to CSV (in main repo root, not dashboard folder)
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_repo_dir = os.path.dirname(script_dir)  # Go up one level from dashboard/ to main repo
    output_file = os.path.join(main_repo_dir, "dashboard_data.csv")
    print(f"\n[5/5] Saving to {output_file}...")
    dashboard_data.to_csv(output_file, index=False)
    print(f"   Saved {len(dashboard_data):,} rows to {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total forecast instances: {len(dashboard_data):,}")
    print(f"Unique storms: {dashboard_data['sid'].nunique():,}")
    print(f"\nBasin distribution:")
    print(dashboard_data['basin_name'].value_counts().head(10))
    print(f"\nStorm type distribution:")
    print(dashboard_data['storm_type'].value_counts().head(10))
    print(f"\nLead times available: {sorted(dashboard_data['lead_time_hours'].unique())}")
    print(f"\nError statistics (km):")
    print(dashboard_data['error_km'].describe())
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nYou can now run the dashboard with: streamlit run app.py")
    
    return dashboard_data

if __name__ == "__main__":
    try:
        dashboard_data = preprocess_dashboard_data()
    except FileNotFoundError as e:
        print("\n" + "=" * 60)
        print("ERROR: Missing Data Files")
        print("=" * 60)
        print(str(e))
        print("\nTo generate the required data files:")
        print("1. Ensure you have run the Kalman Filter evaluation")
        print("2. Check that the data/ directory exists and contains:")
        print("   - hurricane_paths_processed.pkl (or hurricane_paths_processed_MODEL.pkl)")
        print("   - kalman_results_sliding_results_final.pkl (or sliding_results_final.pkl)")
        print("\nIf files are in different locations, update the file paths in this script.")
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR during preprocessing")
        print("=" * 60)
        print(str(e))
        import traceback
        traceback.print_exc()

