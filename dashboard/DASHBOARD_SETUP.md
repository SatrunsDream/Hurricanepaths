# Hurricane Tracking Dashboard - Setup Guide

## Overview

This dashboard provides an interactive "BayesBall-style" exploratory tool for analyzing hurricane track forecast errors. It allows you to:

- **Explore error distributions** by basin, storm type, and lead time
- **Track individual storms** with interactive maps
- **View project methodology** and documentation

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

The dashboard requires two data files:

1. **Track Data** (one of these):
   - `data/hurricane_paths_processed_MODEL.pkl`
   - `data/hurricane_paths_processed.pkl`
   - `hurricane_paths_processed_MODEL.pkl`
   - `hurricane_paths_processed.pkl`

2. **Forecast Results** (one of these):
   - `data/kalman_results_sliding_results_final.pkl`
   - `data/sliding_results_final.pkl`
   - `kalman_results_sliding_results_final.pkl`
   - `sliding_results_final.pkl`

### Step 3: Run Preprocessing

This step merges your track data with forecast results:

```bash
python preprocess_dashboard.py
```

This will create `dashboard_data.csv` which contains:
- Forecast errors at different lead times (6h, 12h, 24h, 48h, 72h)
- Storm metadata (basin, nature/type, season, duration)
- Merged data ready for dashboard filtering

**Expected Output:**
```
HURRICANE TRACKING DASHBOARD - DATA PREPROCESSING
============================================================
[1/3] Loading track data...
   Loaded 721,960 observations from 13,450 storms
[2/3] Extracting storm metadata...
   Extracted metadata for 13,450 storms
[3/3] Loading forecast results...
   Loaded X forecast instances
[4/4] Merging data...
   Successfully merged X forecast instances with metadata
[5/5] Saving to dashboard_data.csv...
   Saved X rows to dashboard_data.csv
```

### Step 4: Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser automatically.

## Dashboard Features

### Tab 1: Scenario Explorer (BayesBall-style)

**Purpose:** Explore forecast error distributions across different scenarios

**Features:**
- Filter by Basin (North Atlantic, Pacific, etc.)
- Filter by Storm Type (Tropical Storm, Hurricane, Extratropical, etc.)
- Select Lead Time (6h, 12h, 24h, 48h, 72h)
- View error distribution histogram with KDE overlay
- See summary metrics: Mean Error, Median Error, RMSE, Max Outlier
- Interpret reliability: Wide curve = High Uncertainty, Narrow spike = Low Uncertainty

**Use Cases:**
- "How reliable is the model for Category 5 storms in the Pacific?"
- "What is the distribution of errors at 48 hours?"
- "Compare error distributions between basins"

### Tab 2: Individual Storm Tracker

**Purpose:** Visualize specific storm tracks and their forecast errors

**Features:**
- Dropdown to select any storm by ID
- Interactive map showing actual storm track
- Error metrics table by lead time
- Error trend plot showing how errors grow with lead time

**Use Cases:**
- "Show me Hurricane Katrina's track and forecast errors"
- "How did the model perform for this specific storm?"

### Tab 3: Methodology

**Purpose:** Display project documentation

**Features:**
- Renders `structure.md` content
- Download button for PDF report (if available)

## Troubleshooting

### Error: "Dashboard data file not found"

**Solution:** Run `python preprocess_dashboard.py` first to create `dashboard_data.csv`

### Error: "Could not find track data file"

**Solution:** 
1. Ensure you have run the feature engineering notebook
2. Check that one of the expected pickle files exists
3. If files are in a different location, update paths in `preprocess_dashboard.py`

### Error: "Could not find forecast results file"

**Solution:**
1. Ensure you have run the Kalman Filter evaluation
2. The `evaluate_sliding_origins()` function should have created the results file
3. Check that the file exists in the `data/` directory

### Dashboard loads but shows "No data matches the selected filters"

**Solution:**
- Check that your forecast results contain data for the selected basin/type/lead time
- Try selecting "All" for basin and storm type filters
- Verify that `dashboard_data.csv` was created successfully

### Map visualization not showing

**Solution:**
- Ensure `hurricane_paths_processed.pkl` (or `hurricane_paths_processed_MODEL.pkl`) exists
- The map requires the full track data file, not just the dashboard CSV

## Data Structure Requirements

### Track Data (`hurricane_paths_processed.pkl`)

Must contain columns:
- `sid` - Storm ID
- `basin` - Basin code (NA, EP, WP, etc.)
- `nature` - Storm nature (TS, ET, HU, etc.)
- `lat`, `lon` - Position coordinates
- `iso_time` - Timestamp
- `season` - Year/season

### Forecast Results (`sliding_results_final.pkl`)

Must contain columns:
- `sid` or `storm_id` - Storm ID (will be renamed to `sid`)
- `lead_time_hours` - Forecast lead time in hours
- `error_km` - Forecast error in kilometers
- `origin_idx` (optional) - Forecast origin index

## Customization

### Adding More Filters

Edit `app.py` Tab 1 section to add additional filters:
- Year/Season filter
- Storm intensity filter
- Duration filter

### Changing Visualization Style

The distribution plot uses Plotly. You can customize:
- Colors in the `fig.add_trace()` calls
- Number of bins in the histogram
- KDE smoothing parameters

### Adding Comparison Features

To compare two scenarios (like "Compare Lead Times" or "Compare Models"):
1. Load multiple result files in preprocessing
2. Add comparison toggle in sidebar
3. Overlay multiple distributions on the same plot

## Performance Notes

- The dashboard uses `@st.cache_data` to cache data loading
- Preprocessing creates a lightweight CSV (~10-50 MB) instead of loading large pickle files
- For very large datasets (>1M rows), consider sampling in preprocessing

## Next Steps

1. **Generate Forecast Results** (if not done):
   - Run `Kalman_Filter.ipynb`
   - Execute `evaluate_sliding_origins()` function
   - Save results using `save_kalman_results()`

2. **Customize Dashboard**:
   - Add your own analysis questions
   - Create custom visualizations
   - Add export functionality

3. **Deploy Dashboard**:
   - Use Streamlit Cloud for free hosting
   - Or deploy on your own server

## Support

If you encounter issues:
1. Check that all required files exist
2. Verify file formats match expected structure
3. Check console output for detailed error messages
4. Review the preprocessing output for warnings

