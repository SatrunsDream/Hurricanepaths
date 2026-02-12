# Streamboard - Hurricane Model Comparison Dashboard

## Overview

Streamboard is a comprehensive Streamlit dashboard for visualizing and comparing hurricane track forecasting model performance, with a focus on the Null Model baseline versus Kalman Filter comparison.

## Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Preprocessing

Generate the optimized CSV files for visualization:

```bash
python preprocess_streamboard.py
```

This creates 5 CSV files:
- `model_comparison_summary.csv` - High-level performance metrics
- `error_distributions.csv` - Detailed error data for plots
- `storm_performance.csv` - Per-storm metrics
- `model_comparison_detailed.csv` - Side-by-side comparisons
- `storm_metadata.csv` - Storm characteristics

### Step 3: Launch Dashboard

```bash
streamlit run app.py
```

## Dashboard Pages

1. **Model Comparison Overview** - High-level performance comparison
2. **Error Distribution Analysis** - Deep dive into error patterns
3. **Storm-Level Performance** - Which storms each model handles better
4. **Individual Storm Tracker** - Visualize specific storm forecasts
5. **Performance by Characteristics** - Performance across basins, types, etc.
6. **Methodology** - Project documentation and context

## Data Requirements

The preprocessing script expects these files in the `data/` directory:

- `hurricane_paths_processed.pkl` - Track data
- `sliding_results_final.pkl` - Kalman filter results
- `null_model_test_results.pkl` - Null model test results
- `null_model_val_results.pkl` - Null model validation results

## Key Findings Highlighted

The dashboard prominently displays the finding that the Null Model (simple persistence) outperforms the Kalman Filter at all lead times, providing essential context for understanding model performance.

## Keeping the App Awake on Streamlit Cloud

Streamlit Community Cloud apps go to sleep after ~1 hour of inactivity. To keep your deployed app always ready:

### Option 1: GitHub Actions (included)

This repo includes `.github/workflows/keep-streamlit-awake.yml` that pings your app every 30 minutes.

**Setup:** Edit the workflow file and replace `YOUR-STREAMLIT-APP-URL` with your actual Streamlit Cloud URL (e.g. `https://hurricanepaths-steamboard-xxxxx.streamlit.app`). Push to GitHub and the workflow will run automatically.

### Option 2: UptimeRobot (no code)

1. Sign up at [uptimerobot.com](https://uptimerobot.com) (free)
2. Add a new HTTP(s) monitor
3. Enter your Streamlit app URL
4. Set check interval to 5 minutes

The periodic requests simulate traffic and prevent the app from sleeping.
