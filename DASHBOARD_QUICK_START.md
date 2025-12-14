# Dashboard Quick Start Guide

## ✅ Current Status

**Good News!** All required data files are already present:
- ✓ `dashboard_data.csv` exists (64.9 KB)
- ✓ Track data files exist (`hurricane_paths_processed.pkl` - 195.3 MB)
- ✓ Forecast results exist (`sliding_results_final.pkl` - 1.9 MB)
- ✓ Dependencies are installed (Streamlit, Pandas, Plotly)

## 🚀 Getting Started (3 Simple Steps)

### Step 1: Verify Dependencies

All required packages are already installed:
- ✅ Streamlit 1.52.1
- ✅ Pandas 2.3.3
- ✅ Plotly 6.5.0
- ✅ NumPy (included with pandas)

**If you need to reinstall:**
```bash
cd dashboard
pip install -r requirements.txt
```

### Step 2: Launch the Dashboard

From the **main project directory** (not the dashboard folder), run:

```bash
streamlit run dashboard/app.py
```

**OR** from inside the dashboard folder:

```bash
cd dashboard
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

### Step 3: Explore!

The dashboard has three tabs:
1. **📊 Scenario Explorer** - Filter by basin, storm type, and lead time to explore error distributions
2. **🗺️ Individual Storm Tracker** - Select specific storms to view tracks and forecast errors
3. **📖 Methodology** - View project documentation

## 📋 What the Dashboard Does

### Tab 1: Scenario Explorer (BayesBall-style)
- **Purpose**: Explore forecast error distributions across different scenarios
- **Features**:
  - Filter by Basin (North Atlantic, Pacific, etc.)
  - Filter by Storm Type (Tropical Storm, Hurricane, etc.)
  - Select Lead Time (6h, 12h, 24h, 48h, 72h)
  - View error distribution histogram with KDE overlay
  - See summary metrics: Mean Error, Median Error, RMSE, Max Outlier
  - Reliability assessment based on coefficient of variation

### Tab 2: Individual Storm Tracker
- **Purpose**: Visualize specific storm tracks and their forecast errors
- **Features**:
  - Dropdown to select any storm by ID
  - Interactive map showing actual storm track
  - Error metrics table by lead time
  - Error trend plot showing how errors grow with lead time

### Tab 3: Methodology
- **Purpose**: Display project documentation
- **Features**:
  - Renders `structure.md` content
  - Download button for PDF report (if available)

## 🔧 Troubleshooting

### Issue: "Dashboard data file not found"
**Solution**: The `dashboard_data.csv` file already exists. If you see this error:
1. Make sure you're running from the correct directory
2. Check that `dashboard_data.csv` is in the main project root
3. If missing, run: `python dashboard/preprocess_dashboard.py`

### Issue: "Could not find track data file"
**Solution**: Track data files exist. If you see this error:
- The dashboard looks for files in `data/hurricane_paths_processed.pkl` or `data/hurricane_paths_processed_MODEL.pkl`
- Both files exist, so this shouldn't happen
- If it does, check file permissions

### Issue: Map visualization not showing
**Solution**: 
- Ensure `data/hurricane_paths_processed.pkl` exists (it does - 195.3 MB)
- The map requires the full track data file, not just the dashboard CSV
- Check browser console for JavaScript errors

### Issue: "No data matches the selected filters"
**Solution**:
- Try selecting "All" for basin and storm type filters
- Check that `dashboard_data.csv` contains data for your selected filters
- Verify the CSV file is not corrupted

### Issue: Port already in use
**Solution**: 
```bash
# Use a different port
streamlit run dashboard/app.py --server.port 8502
```

## 📊 Data Files Overview

### Required Files (All Present ✅)

1. **`dashboard_data.csv`** (64.9 KB)
   - Preprocessed data merging track metadata with forecast errors
   - Contains: storm IDs, lead times, errors, basin, storm type, etc.
   - Created by: `dashboard/preprocess_dashboard.py`

2. **`data/hurricane_paths_processed.pkl`** (195.3 MB)
   - Full processed hurricane track data
   - Used for: Map visualization in Tab 2
   - Contains: lat, lon, iso_time, sid, and other track features

3. **`data/sliding_results_final.pkl`** (1.9 MB)
   - Forecast error results from Kalman filter evaluation
   - Contains: storm IDs, lead times, forecast errors
   - Used to create: `dashboard_data.csv`

### Optional Files

- `data/val_sliding_results_final.pkl` - Validation set results (if you want to include validation data)
- `data/kalman_results_sliding_results_final.pkl` - Alternative results format

## 🔄 Regenerating Dashboard Data

If you need to regenerate `dashboard_data.csv` (e.g., after updating forecast results):

```bash
python dashboard/preprocess_dashboard.py
```

This script will:
1. Load track data from `data/hurricane_paths_processed.pkl`
2. Load forecast results from `data/sliding_results_final.pkl` (or validation set)
3. Merge them together with storm metadata
4. Save to `dashboard_data.csv` in the main project root

## 📝 File Structure

```
hurricanepaths/
├── dashboard/
│   ├── app.py                    # Main dashboard application
│   ├── preprocess_dashboard.py   # Data preprocessing script
│   ├── requirements.txt          # Python dependencies
│   └── DASHBOARD_SETUP.md        # Detailed setup guide
├── data/
│   ├── hurricane_paths_processed.pkl      # Track data (195.3 MB)
│   ├── sliding_results_final.pkl          # Forecast results (1.9 MB)
│   └── val_sliding_results_final.pkl      # Validation results (1.5 MB)
├── dashboard_data.csv            # Preprocessed dashboard data (64.9 KB)
└── structure.md                  # Project methodology (displayed in Tab 3)
```

## 🎯 Quick Test

To verify everything works:

1. **Start the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

2. **Test Tab 1 (Scenario Explorer):**
   - Select "All" for Basin
   - Select "All" for Storm Type  
   - Select "24" hours for Lead Time
   - You should see an error distribution plot

3. **Test Tab 2 (Individual Storm Tracker):**
   - Select any storm from the dropdown
   - You should see storm metadata and error metrics
   - Map visualization should appear (if track data loads)

4. **Test Tab 3 (Methodology):**
   - Should display content from `structure.md`

## 💡 Tips

- **Performance**: The dashboard uses caching (`@st.cache_data`) so data loads quickly after the first load
- **Filtering**: Use the sidebar filters to explore different scenarios
- **Export**: You can export filtered data using Streamlit's built-in download features
- **Customization**: Edit `dashboard/app.py` to add more filters or visualizations

## 🆘 Need Help?

1. Check `dashboard/DASHBOARD_SETUP.md` for detailed troubleshooting
2. Review the preprocessing output: `python dashboard/preprocess_dashboard.py`
3. Check file paths in `dashboard/app.py` if files are in different locations
4. Verify Python environment has all dependencies installed

---

**You're all set!** Just run `streamlit run dashboard/app.py` and start exploring your hurricane forecast data! 🌀

