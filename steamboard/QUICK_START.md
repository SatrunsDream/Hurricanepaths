# Steamboard Quick Start Guide

## 🚀 Launch the Dashboard

### Step 1: Ensure Data Files Are Ready

The preprocessing script should have already created these CSV files:
- ✅ `model_comparison_summary.csv`
- ✅ `error_distributions.csv`
- ✅ `storm_performance.csv`
- ✅ `model_comparison_detailed.csv`
- ✅ `storm_metadata.csv`

If not, run:
```bash
cd steamboard
python preprocess_steamboard.py
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## 📊 Dashboard Features

### Page 1: Model Comparison Overview
- High-level RMSE and mean error comparison
- Key finding highlight (Null Model outperforms KF)
- Performance improvement metrics
- Detailed statistics table

### Page 2: Error Distribution Analysis
- Side-by-side histograms
- Cumulative distribution curves
- Box plot comparisons
- Error category breakdowns
- Filters: Lead time, model, basin, storm type

### Page 3: Storm-Level Performance
- Scatter plot: KF vs Null error
- Better model distribution pie chart
- Performance by basin
- Storm ranking table
- Filters: Basin, storm type, error threshold

### Page 4: Individual Storm Tracker
- Error trend plots for selected storms
- Error metrics by lead time
- Storm metadata display

### Page 5: Performance by Characteristics
- Performance by basin (grouped bars)
- Performance by storm type
- RMSE heatmaps (basin × lead time)

### Page 6: Methodology
- Project documentation
- Key findings
- Evaluation methodology
- Data processing pipeline

## 🎨 Design Features

- **Modern UI**: Gradient headers, clean styling, professional layout
- **Interactive Charts**: Plotly visualizations with hover tooltips
- **Color Scheme**: 
  - Red (#e74c3c) for Null Model
  - Blue (#3498db) for Kalman Filter
- **Responsive**: Wide layout optimized for large screens
- **Fast Loading**: Cached data loading for performance

## 💡 Tips

1. **Use Filters**: Each page has filters to explore specific scenarios
2. **Hover for Details**: All charts have interactive hover tooltips
3. **Compare Models**: Use side-by-side visualizations to see differences
4. **Explore Storms**: Use the Individual Storm Tracker to dive deep into specific storms

## 🔧 Troubleshooting

**Dashboard won't start:**
- Check that all CSV files exist in the `steamboard` folder
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Verify Python version (3.8+)

**Data not loading:**
- Run `python preprocess_steamboard.py` to regenerate CSV files
- Check that data files exist in `../data/` directory

**Charts not displaying:**
- Check browser console for errors
- Ensure Plotly is installed: `pip install plotly>=5.14.0`

## 📝 Notes

- The dashboard uses cached data loading for performance
- All visualizations are interactive Plotly charts
- Data is filtered client-side for fast interactions
- The dashboard highlights the key finding: Null Model outperforms Kalman Filter
