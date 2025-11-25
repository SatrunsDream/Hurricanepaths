# Project Structure and Progress Documentation

## Project Overview

**Project**: Hurricane Track Prediction using Kalman Filter  
**Dataset**: IBTrACS (International Best Track Archive for Climate Stewardship)  
**Objective**: Implement a probabilistic state-space model for hurricane track forecasting using the Kalman filter

---

## File Structure

### Data Files
- `ibtracs.ALL.list.v04r01.csv` - Main dataset containing hurricane track data from 1842-2025
- `dataset_summary.json` - Structured metadata summary generated during EDA (722K observations, 174 columns, 13.5K storms)
- `hurricane_paths_processed.pkl` - Processed dataset ready for Kalman filter (721,960 observations, 13,450 storms)
- `hurricane_paths_processed.csv` - Processed dataset in CSV format

### Analysis Notebooks
- `eda_cleaning.ipynb` - Exploratory Data Analysis and data cleaning (COMPLETED)
- `features_engineering.ipynb` - Feature engineering for Kalman filter (COMPLETED)

### Documentation
- `README.md` - Project overview and setup instructions
- `structure.md` - This file: comprehensive project structure and progress tracking
- `Report/CSE150A Milestone 1 Report.pdf` - Project proposal and methodology
- `References/FORECASTING HURRICANE TRACKS USING THE KALMAN FILTER.pdf` - Reference paper

---

## Phase 1: Exploratory Data Analysis (COMPLETED)

### File: `eda_cleaning.ipynb`

#### Objectives
1. Understand dataset structure and contents
2. Assess data quality and completeness
3. Identify key variables for Kalman filter state-space model
4. Validate temporal structure (6-hour intervals)
5. Document data characteristics and units

#### Cells Breakdown

**Cell 0: Header**
- Project goal statement
- Data documentation link

**Cell 1: Imports**
- pandas, numpy, matplotlib, seaborn
- Display configuration

**Cell 2: Data Loading Function**
- `load_ibtracs()` function definition
- Handles IBTrACS two-header format
- Normalizes column names
- Replaces blank values with NaN
- Type conversions for mixed-type columns
- Datetime conversion for iso_time
- Loads full dataset: `hurricane_paths`

**Cell 3-4: Initial Exploration**
- Basic dtype inspection
- Sample data viewing

**Cell 5: Dataset Overview**
- Dataset shape: 722,040 rows × 174 columns
- Complete column listing

**Cell 6: Column-by-Column Analysis**
- Data type identification
- Null/non-null counts
- Numeric statistics (min, max, mean, median)
- Unique value enumeration for categorical variables
- Sample values for high-cardinality columns

**Cell 7: Sample Data Display**
- First 5 rows of dataset

**Cell 8: Example Storm Track**
- Single storm track visualization
- Temporal structure verification

**Cell 9: Dataset Summary Creation**
- `create_dataset_summary()` function
- Generates comprehensive JSON metadata
- Categorizes columns (position, velocity, wind, pressure)
- Analyzes data source coverage
- Saves to `dataset_summary.json`

**Cell 10: Key Columns Summary**
- Position columns overview
- Velocity/motion columns analysis
- Wind speed columns summary

**Cell 11: Temporal Structure Analysis**
- Time interval verification (6-hour intervals)
- Observations per storm statistics
- Interval distribution analysis

**Cell 12: Sample Storm Track Deep Dive**
- Selects well-documented storm (HELINDA:PANCHO 1997)
- Detailed track analysis
- Position, velocity, wind statistics
- Stores sample for visualization

**Cell 13: Data Quality Analysis**
- Missing velocity data by storm
- Position data bounds validation
- Velocity data range checks
- Position jump detection (potential errors)

**Cell 14: [Empty - Reserved for Summary]**
- Could be used for additional summary or visualization

**Cell 15: Pre-Feature Engineering Assessment**
- Storm length distribution analysis
- Velocity computation validation
- Basin distribution overview
- Temporal coverage by decade
- Unit and coordinate system specification
- **Conclusion**: Ready for feature engineering

**Cell 16: Basin Distribution Analysis**
- Basin statistics
- North Atlantic region identification
- Basin code conventions

---

## Key Findings from EDA

### Dataset Characteristics
- **Size**: 722,040 observations across 13,530 unique storms
- **Temporal Range**: 1842-10-25 to 2025-11-23
- **Columns**: 174 total columns
- **Position Data**: 100% coverage (lat/lon)
- **Velocity Data**: 99.99% coverage (storm_speed, storm_dir)

### Data Quality
- **Missing Velocity**: Only 80 storms (0.59%), all single-observation storms
- **Storms with ≥2 observations**: 13,450 (99.4%)
- **Storms with ≥5 observations**: 13,420 (99.2%)
- **Storms with ≥10 observations**: 13,053 (96.5%)
- **Temporal Structure**: Regular 6-hour observation intervals confirmed

### Available Variables

**State Variables (for Kalman Filter)**
- Position: `lat`, `lon` (best track, 100% coverage)
- Velocity: `storm_speed` (knots), `storm_dir` (degrees, 0-360)

**Observation Variables**
- Wind speed: Multiple sources (usa_wind, wmo_wind, tokyo_wind, etc.)
- Pressure: Multiple sources (usa_pres, wmo_pres, etc.)
- Position: Same as state (observed lat/lon)

**Additional Features**
- Distance to land: `dist2land` (km)
- Storm nature: `nature` (TS, ET, MX, etc.)
- Basin information: `basin`, `subbasin`
- Intensity measures: Multiple sources available

### Basin Distribution
- **WP** (Western Pacific): 241,388 observations
- **SI** (South Indian): 162,487 observations
- **SP** (South Pacific): 68,076 observations
- **EP** (Eastern Pacific): 65,391 observations
- **NI** (North Indian): 57,532 observations
- **SA** (South Atlantic): 119 observations

### Units Specification
- **Position**: Degrees (lat: -90 to 90, lon: -180 to 180)
- **Velocity**: Speed in knots, direction in degrees (0-360)
- **Time**: 6-hour intervals
- **Wind Speed**: Knots (various sources)

---

## Phase 2: Feature Engineering (COMPLETED)

### File: `features_engineering.ipynb`

The feature engineering phase transforms the cleaned IBTrACS dataset into a format suitable for Kalman filter implementation. The process begins by carrying over the data cleaning function from the EDA phase, which handles the IBTrACS two-header format, normalizes column names, replaces blank values with NaN, performs type conversions for mixed-type columns, and converts timestamps to datetime format.

Data filtering was implemented to ensure only storms with sufficient observations for velocity computation are included. The dataset is filtered to storms with at least two observations, as velocity requires position differences between consecutive time steps. Single-observation storms are excluded, resulting in a filtered dataset of 13,450 valid storms from the original 13,530.

Velocity computation addresses the minimal missing velocity data by implementing a physics-based approach. A function was created to compute velocity from position differences using haversine distance calculations. This function handles longitude wrapping at ±180 degrees, converts geographic coordinates to kilometers accounting for latitude-dependent longitude scaling, and computes both speed and direction. Missing velocity values are filled using this computational method rather than statistical imputation, as the missing data is minimal (0.01%) and occurs only in storms that now have sufficient observations after filtering.

The velocity representation was converted to Cartesian components suitable for linear Kalman filter operations. The speed and direction values are transformed into velocity components (v_lat, v_lon) measured in degrees per 6-hour interval. This conversion accounts for the spherical geometry of Earth's surface, with longitude velocity adjusted by the cosine of latitude to maintain proper distance calculations. The state vector is now represented as [latitude, longitude, v_lat, v_lon], which enables linear state-space modeling.

Temporal features were extracted to capture climatological and seasonal effects that may influence storm behavior. Storm age is computed as the hours elapsed since the first observation of each storm. Additional temporal features include day of year and month, which can be used to model seasonal patterns in hurricane tracks. These features follow the approach used in operational forecasting models like T-CLIPER and SHIFOR, which incorporate climatological information.

Acceleration features were computed to enhance the state representation. Acceleration is calculated as the change in velocity components over time, providing information about storm motion dynamics beyond simple velocity. This follows trajectory-based modeling approaches similar to the TAB (Trajectory and Beta) model used by the National Hurricane Center, which considers both advection and beta effects in tropical cyclone motion.

Additional advanced features were engineered to enhance Kalman filter performance and adaptability. Track curvature was computed to measure how sharply storms are turning, which helps identify when the linear motion assumption breaks down. Latitude regime classification categorizes storms into tropics, subtropics, and mid-latitudes, as storm motion characteristics vary significantly by latitude. Hemisphere indicator and motion regime classification distinguish between westward, poleward/recurving, and low-motion patterns, enabling adaptive model parameters. Storm stage encoding captures the developmental phase of each storm (disturbance, depression, tropical storm, hurricane, extratropical), which affects motion characteristics. Landfall proximity features include both a binary flag for storms within 200 km of land and a land gradient feature tracking the rate of approach to land, as storms behave differently when near coastlines. Beta-drift proxy features approximate Coriolis-related drift effects that influence tropical cyclone motion. Smoothed velocity features using 3-point moving averages reduce observational noise. Autoregressive motion features capture recent motion trends through 6-hour and 12-hour averages, providing context for motion persistence.

The final processed dataset contains key state variables including position (lat, lon), velocity components (v_lat, v_lon), original velocity representation (storm_speed, storm_dir), acceleration components (a_lat, a_lon), temporal features, advanced regime and classification features, and metadata (basin, nature). The dataset was validated to ensure no missing values in critical state variables, resulting in 721,960 observations across 13,450 unique storms spanning from 1842 to 2025. The processed dataset is saved in both pickle format (preserving dtypes) and CSV format for compatibility.

**Key Outcomes:**
- Filtered from 13,530 to 13,450 storms (removed 80 single-observation storms)
- Final dataset: 721,960 observations with zero missing values in state variables
- State vector format: [lat, lon, v_lat, v_lon] ready for Kalman filter
- Velocity components computed in degrees per 6-hour interval
- Acceleration features computed for enhanced state representation
- Temporal features included for climatological modeling
- Advanced features added: track curvature, latitude/motion regimes, landfall proximity, beta-drift proxy, smoothed velocities, autoregressive motion features

---

## Phase 3: Kalman Filter Implementation (FUTURE)

### Planned Components

### Planned Components

#### State-Space Model Design
- **State Vector (x_t)**: [latitude, longitude, velocity_lat, velocity_lon] or equivalent
- **Observation Vector (y_t)**: [observed_lat, observed_lon] (possibly with intensity measures)
- **Transition Matrix (A)**: Linear dynamics model
- **Observation Matrix (H)**: Maps state to observations
- **Process Noise Covariance (Q)**: Uncertainty in storm motion
- **Observation Noise Covariance (R)**: Uncertainty in position measurements

#### Kalman Filter Algorithm
1. **Initialization**: Initial state estimate and covariance
2. **Prediction Step**: Predict state at next time step
3. **Update Step**: Incorporate new observation
4. **Forecast Generation**: One-step-ahead predictions

#### Evaluation
- Forecast error computation
- Comparison with actual best-track positions
- Error statistics (mean error, RMSE, etc.)
- Visualization of forecast tracks

---

## Data Flow Summary

```
Raw Data (ibtracs.ALL.list.v04r01.csv)
    ↓
EDA & Cleaning (eda_cleaning.ipynb)
    ├── Data loading and normalization
    ├── Quality assessment
    ├── Structure understanding
    └── Summary generation (dataset_summary.json)
    ↓
Feature Engineering (features_engineering.ipynb) [COMPLETED]
    ├── Data filtering (storms with ≥2 observations)
    ├── Velocity computation from positions
    ├── Cartesian velocity components (v_lat, v_lon)
    ├── Acceleration features
    ├── Temporal features (storm age, seasonality)
    └── Processed dataset saved (hurricane_paths_processed.pkl/csv)
    ↓
Kalman Filter Implementation [FUTURE]
    ├── Model design
    ├── Parameter estimation
    ├── Filter implementation
    └── Forecast evaluation
```

---

## Key Decisions and Assumptions

### Data Source Selection
- **Position**: Use `lat`, `lon` (best track, 100% coverage)
- **Velocity**: Use `storm_speed`, `storm_dir` (99.99% coverage)
- **Observations**: Can use same position fields or select specific agency data

### Filtering Decisions
- Minimum storm length: ≥2 observations (required for velocity)
- Recommended: ≥5 observations for Kalman filter
- Optional: Focus on specific basin for consistency

### Coordinate System
- Currently in degrees (latitude/longitude)
- May need conversion to metric for some implementations
- Consider spherical geometry for accurate distance calculations

---

## Next Steps

1. **Create `feature_engineering.ipynb`**
   - Start with data filtering
   - Implement velocity computation function
   - Standardize units

2. **Design State-Space Model**
   - Define state vector representation
   - Design transition model
   - Design observation model

3. **Estimate Parameters**
   - Estimate process noise covariance Q
   - Estimate observation noise covariance R
   - Tune model parameters

4. **Implement Kalman Filter**
   - Write filter algorithm
   - Test on sample storms
   - Validate predictions

5. **Evaluate and Visualize**
   - Compute forecast errors
   - Generate "spaghetti" plots
   - Compare with benchmark models

---

## Notes and Considerations

- The dataset spans 183 years, quality may vary by era
- Multiple data sources available (USA, WMO, Tokyo, etc.) - need to select or combine
- Some storms may have irregular intervals (need to handle)
- Direction values wrap around (0-360 degrees) - handle carefully
- Longitude values may exceed ±180 in some datasets (check handling)

---

## References

- IBTrACS Documentation: https://www.ncei.noaa.gov/sites/g/files/anmtlf171/files/2025-09/IBTrACS_v04r01_column_documentation.pdf
- Bril, G. (1995). Forecasting hurricane tracks using the Kalman filter. Environmetrics, 6(1), 7-16.
- Project Proposal: `Report/CSE150A Milestone 1 Report.pdf`

