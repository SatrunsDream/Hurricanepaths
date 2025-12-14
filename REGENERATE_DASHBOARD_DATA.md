# How to Regenerate Dashboard Data

## Quick Steps to Regenerate dashboard_data.csv

Since you've updated the data files in the `data/` folder, you need to regenerate `dashboard_data.csv` to reflect the new data.

### Step 1: Navigate to Project Root

Make sure you're in the main project directory:
```bash
cd C:\Users\sardo\OneDrive\Desktop\Classes\CSE150A\hurricanepaths
```

### Step 2: Run the Preprocessing Script

Run the preprocessing script that will:
1. Load your updated track data
2. Load your updated forecast results
3. Merge them together
4. Create a new `dashboard_data.csv`

**Command:**
```bash
python dashboard/preprocess_dashboard.py
```

### Step 3: Verify the Output

The script will print progress information. You should see:
```
============================================================
HURRICANE TRACKING DASHBOARD - DATA PREPROCESSING
============================================================

[1/4] Loading track data...
   Loaded X observations from Y storms
[2/4] Extracting storm metadata...
   Extracted metadata for Y storms
[3/4] Loading forecast results...
   Loaded X forecast instances
[4/4] Merging data...
   Successfully merged X forecast instances with metadata
[5/5] Saving to dashboard_data.csv...
   Saved X rows to dashboard_data.csv
```

### Step 4: Check the New File

After completion, verify the new file was created:
```bash
python check_setup.py
```

Or manually check:
- File: `dashboard_data.csv` should exist in the project root
- File size should be reasonable (typically 50-500 KB depending on data)

## What Files Are Used

The preprocessing script looks for these files (in order of priority):

### Track Data (one of these):
- `data/hurricane_paths_processed.pkl` (preferred)
- `data/hurricane_paths_processed_MODEL.pkl`

### Forecast Results (one or both):
- `data/sliding_results_final.pkl` (test set - preferred)
- `data/kalman_results_sliding_results_final.pkl` (alternative format)
- `data/val_sliding_results_final.pkl` (validation set - optional, will be combined if found)

## Troubleshooting

### Error: "Could not find track data file"
**Solution:** Make sure your track data file exists:
- Check: `data/hurricane_paths_processed.pkl` exists
- Or: `data/hurricane_paths_processed_MODEL.pkl` exists

### Error: "Could not find forecast results file"
**Solution:** Make sure your results file exists:
- Check: `data/sliding_results_final.pkl` exists
- Or: `data/kalman_results_sliding_results_final.pkl` exists

### Error: "No matching storm IDs found"
**Solution:** This means the storm IDs in your track data don't match the storm IDs in your forecast results. Check:
- Both files use the same storm ID format (should be `sid` column)
- Both files contain data for the same storms
- The forecast results were generated from the same track data

### Warning: "Only X% of forecasts matched with storm metadata"
**Solution:** This is usually okay if >90% match. If less, check:
- Storm IDs match between files
- Both files contain overlapping storms

## After Regeneration

Once `dashboard_data.csv` is regenerated:

1. **Start the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

2. **Clear browser cache** (if dashboard shows old data):
   - Press `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac) to hard refresh
   - Or click the hamburger menu → "Clear cache" → "Rerun"

## Full Command Sequence

Here's everything in one go:

```bash
# 1. Make sure you're in the project root
cd C:\Users\sardo\OneDrive\Desktop\Classes\CSE150A\hurricanepaths

# 2. Regenerate dashboard data
python dashboard/preprocess_dashboard.py

# 3. Verify it worked
python check_setup.py

# 4. Start dashboard
streamlit run dashboard/app.py
```

## Expected Output

When successful, you'll see output like:

```
============================================================
HURRICANE TRACKING DASHBOARD - DATA PREPROCESSING
============================================================

[1/4] Loading track data...
Loading track data from: C:\Users\sardo\...\data\hurricane_paths_processed.pkl
   Loaded 721,960 observations from 13,450 storms

[2/4] Extracting storm metadata...
   Extracted metadata for 13,450 storms
   Metadata columns: ['sid', 'basin', 'nature', 'start_time', 'end_time', ...]

[3/4] Loading forecast results...
Loading test set results from: C:\Users\sardo\...\data\sliding_results_final.pkl
   Loaded 2,780 forecast instances

[4/4] Merging data...
   Results file has 556 unique storm IDs
   Metadata has 13,450 unique storm IDs
   Matching storm IDs: 556
   Successfully merged 2,780 forecast instances with metadata

[5/5] Saving to dashboard_data.csv...
   Saved 2,780 rows to dashboard_data.csv

============================================================
DATA SUMMARY
============================================================
Total forecast instances: 2,780
Unique storms: 556
Basin distribution: ...
Storm type distribution: ...
Lead times available: [6, 12, 24, 48, 72]

============================================================
PREPROCESSING COMPLETE!
============================================================

You can now run the dashboard with: streamlit run app.py
```

---

**That's it!** After running `python dashboard/preprocess_dashboard.py`, your `dashboard_data.csv` will be updated with the latest data from your `data/` folder.

