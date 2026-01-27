# Streamboard Dashboard Layout Design

## Overview

The Streamboard dashboard provides comprehensive visualization and analysis of hurricane track forecasting model performance, with a focus on comparing the Null Model baseline against the Kalman Filter implementation.

## Dashboard Structure

### Page 1: Model Comparison Overview
**Purpose**: High-level comparison of Null Model vs Kalman Filter performance

**Key Visualizations**:
- RMSE comparison line chart (by lead time) - showing Null Model outperforms KF
- Mean error comparison bar chart
- Performance improvement/degradation metrics
- Summary statistics table

**Filters**:
- Dataset selection (test/validation/all)
- Lead time selector

**Key Metrics Display**:
- Overall RMSE by model
- Improvement percentage at each lead time
- Number of forecast instances

---

### Page 2: Error Distribution Analysis
**Purpose**: Deep dive into error distributions and patterns

**Key Visualizations**:
- Side-by-side error distribution histograms (Null vs KF) by lead time
- Cumulative error distribution curves
- Box plots comparing error distributions
- Error category breakdown (Excellent/Good/Fair/Poor/Very Poor)

**Filters**:
- Lead time selector
- Model selector (both/individual)
- Basin filter
- Storm type filter

**Insights**:
- Show that Null Model has tighter error distributions
- Highlight where each model performs better
- Error growth patterns with lead time

---

### Page 3: Storm-Level Performance
**Purpose**: Explore which storms each model handles better

**Key Visualizations**:
- Scatter plot: KF error vs Null error (with diagonal reference)
- Storm performance ranking table
- Basin-specific performance comparison
- Storm type performance comparison

**Filters**:
- Basin selector
- Storm type selector
- Performance threshold slider
- Sort by various metrics

**Features**:
- Clickable storms to drill down
- Highlight storms where models differ significantly
- Show distribution of "better model" assignments

---

### Page 4: Individual Storm Tracker
**Purpose**: Visualize specific storm tracks and forecast comparisons

**Key Visualizations**:
- Interactive map showing actual track
- Overlay of Null Model forecast track
- Overlay of Kalman Filter forecast track
- Error metrics table by lead time
- Error trend plot (both models)

**Features**:
- Storm selector dropdown
- Forecast origin selector
- Lead time selector for forecast visualization
- Download storm data option

---

### Page 5: Performance by Characteristics
**Purpose**: Understand model performance across different storm characteristics

**Key Visualizations**:
- Performance by basin (grouped bar chart)
- Performance by storm type (grouped bar chart)
- Performance by storm length/duration
- Performance by season/year
- Heatmap: RMSE by basin × lead time

**Filters**:
- Model selector
- Lead time selector
- Characteristic selector

**Insights**:
- Identify conditions where Null Model advantage is largest
- Find scenarios where KF might perform better
- Understand systematic patterns

---

### Page 6: Methodology & Documentation
**Purpose**: Display project methodology and context

**Content**:
- Render structure.md content
- Key findings summary
- Model descriptions
- Evaluation methodology
- Data sources and processing steps
- PDF report download link

**Sections**:
- Project Overview
- Data Processing Pipeline
- Model Implementations
- Evaluation Framework
- Key Findings
- References

---

## Data Files Created by Preprocessing

1. **model_comparison_summary.csv**
   - One row per lead time
   - Columns: lead_time_hours, kf_mean_error, kf_rmse, null_mean_error, null_rmse, improvement_pct, etc.
   - Use for: Overview page, summary metrics

2. **error_distributions.csv**
   - One row per forecast instance
   - Columns: sid, lead_time_hours, error_km, model, error_category, etc.
   - Use for: Distribution plots, histograms, cumulative distributions

3. **storm_performance.csv**
   - One row per storm
   - Columns: sid, kf_mean_error, null_mean_error, better_model, basin, storm_type, etc.
   - Use for: Storm-level analysis, scatter plots, rankings

4. **model_comparison_detailed.csv**
   - One row per forecast instance (matched)
   - Columns: sid, origin_idx, lead_time_hours, kf_error_km, null_error_km, error_difference, null_better
   - Use for: Side-by-side comparisons, improvement analysis

5. **storm_metadata.csv**
   - One row per storm
   - Columns: sid, basin, basin_name, nature, storm_type, season, duration_days, observation_count, etc.
   - Use for: Filtering, grouping, metadata display

---

## Design Principles

1. **Highlight Key Finding**: Null Model outperforms Kalman Filter - make this prominent
2. **Interactive Exploration**: Allow users to drill down into specific scenarios
3. **Clear Comparisons**: Side-by-side visualizations where possible
4. **Contextual Information**: Always show what filters are active and sample sizes
5. **Performance Metrics**: Consistent use of RMSE, mean error, median error
6. **Visual Hierarchy**: Most important findings at the top, details below

---

## Technical Considerations

- Use Plotly for interactive charts
- Streamlit caching for data loading
- Efficient filtering and aggregation
- Responsive layout (wide mode)
- Color scheme: Red for Null Model, Blue for Kalman Filter (or vice versa)
- Consistent styling across all pages

---

## Future Enhancements

- Add more models if available
- Export functionality for filtered data
- Statistical significance testing
- Forecast trajectory animations
- Ensemble forecast visualization
