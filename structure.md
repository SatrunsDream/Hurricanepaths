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
- `Kalman_Filter.ipynb` - Kalman filter implementation and evaluation (COMPLETED)

### Documentation
- `README.md` - Project overview and setup instructions
- `structure.md` - This file: comprehensive project structure and progress tracking
- `Report/CSE150A Report.pdf` - Project methodology
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

## Phase 3: Kalman Filter Implementation (COMPLETED)

### File: `Kalman_Filter.ipynb`

#### Overview

The Kalman Filter implementation provides a probabilistic state-space model for hurricane track prediction using sequential Bayesian inference. The implementation includes a feature-adaptive Kalman filter that adjusts model parameters based on storm characteristics, enabling more accurate predictions for different storm regimes and motion patterns.

#### Data Preparation and Train/Test Split

The processed dataset from feature engineering is loaded, containing 721,960 observations from 13,450 storms with metric coordinates. The data is split at the storm level (80/20) to ensure no data leakage between training and testing phases. This results in 10,759 training storms (577,711 observations) and 2,690 test storms (144,247 observations).

Storms are filtered to include only those with at least 3 observations to enable proper Kalman filter initialization and evaluation. All storms in the dataset use metric coordinates (x_km, y_km, vx_km, vy_km) which enable linear state-space modeling.

#### State-Space Model Design

The Kalman filter uses a constant velocity model with the following specifications:

**State Vector (x_t)**: [x_km, y_km, vx_km, vy_km] in metric coordinates
- x_km: east-west position in km (relative to storm start)
- y_km: north-south position in km (relative to storm start)  
- vx_km: east-west velocity in km per 6 hours
- vy_km: north-south velocity in km per 6 hours

**Observation Vector (y_t)**: [x_km, y_km] - observed positions only

**Transition Matrix (A)**: Implements constant velocity dynamics
- Position updates linearly with velocity: x_{t+1} = x_t + vx, y_{t+1} = y_t + vy
- Velocities remain constant: vx_{t+1} = vx_t, vy_{t+1} = vy_t

**Observation Matrix (H)**: Maps state to observations by selecting position components only

#### HurricaneKalmanFilter Class Implementation

The core filter class implements the standard Kalman filter algorithm with several enhancements:

**Core Methods:**
- `initialize()`: Sets initial state estimate and covariance matrix
- `predict()`: Forecasts next state using transition dynamics and process noise
- `update()`: Incorporates new observation using Kalman gain to update state and covariance
- `forecast()`: Generates multi-step ahead forecasts for probabilistic forecasting

**Feature-Adaptive Parameters:**

The filter implements adaptive process noise (Q) scaling based on storm features, which allows the model to account for increased uncertainty during specific conditions:

- Track curvature: Higher Q (up to 3×) when storms are turning sharply, indicating breakdown of linear motion assumption
- Land approach: 1.5× Q scaling when storms are within 200 km of land, as storms behave unpredictably near coastlines
- Motion regimes: Different Q scaling for westward (1.1×), poleward/recurving (1.3×), and low-motion (1.0×) patterns
- Latitude regimes: 1.2× Q scaling in mid-latitudes where storms interact with extratropical systems

This adaptive approach enables the filter to dynamically adjust uncertainty estimates based on storm characteristics, improving prediction accuracy across diverse storm conditions.

#### Parameter Estimation from Training Data

Process noise covariance (Q) and observation noise covariance (R) are estimated directly from training data using a sample of 200 storms with at least 5 observations each.

**Process Noise Q Estimation:**
Q is estimated by computing the covariance of innovations (differences between predicted and actual state transitions) under the constant velocity model. This captures the inherent uncertainty in storm motion dynamics.

**Observation Noise R Estimation:**
R is estimated from observation residuals (innovation) during filtering. The estimation uses actual filtered states versus observed positions, properly capturing best-track uncertainty including coordinate conversion noise. The estimated R has a minimum threshold of 0.25 km² variance to account for best-track measurement uncertainty (approximately 0.5 km standard deviation).

**Estimated Parameters (from training sample):**
- Q (process noise): Large covariance values reflecting high uncertainty in storm motion dynamics, with significant cross-correlations between position and velocity components
- R (observation noise): Estimated at approximately 264.92 km² variance for x-coordinate and 168.61 km² variance for y-coordinate, reflecting realistic best-track uncertainty

#### Filtering Implementation

The `run_kalman_filter_on_storm()` function applies the Kalman filter to a single storm track with optional feature adaptation. The function initializes the filter with the first observation, then sequentially performs prediction and update steps for each subsequent observation. Feature adaptation occurs before each prediction step, allowing the filter to adjust parameters dynamically as the storm evolves.

The function returns filtered states, one-step-ahead predictions, observations, covariance matrices, and reference coordinates for visualization. This implementation evaluates filtering accuracy (tracking with updates), which provides an upper bound on model performance.

#### Evaluation Results

**Single Storm Test:**
A test storm with 65 observations demonstrated filtering accuracy with mean forecast error of 10.15 km, RMSE of 13.37 km, and median error of 8.25 km. These results represent one-step-ahead prediction accuracy with full observations available at each step.

**Test Set Evaluation (Filtering Mode):**
Evaluation on 2,680 test storms with at least 5 observations shows:
- Mean forecast error: 21.13 km
- RMSE: 26.85 km  
- Median error: 16.85 km

Error distribution statistics indicate consistent performance across diverse storms, with the majority of storms showing errors between 13.79 km and 25.14 km (interquartile range). Maximum errors reach 203 km, likely corresponding to storms with rapid motion changes or unusual track patterns.

#### Open-Loop Forecasting Implementation

A critical distinction was made between filtering accuracy (evaluating the filter's ability to track storms with continuous updates) and true forecasting accuracy (evaluating predictions without future observations). Two new functions were implemented for proper open-loop forecasting evaluation:

**`initialize_filter_from_observations()`:**
Initializes the filter using observations up to a specified time index (t0), then returns the filter in its state at that time. This provides the starting point for open-loop forecasting where no future observations are used.

**`open_loop_forecast()`:**
Generates true open-loop forecasts by initializing from observations up to t0, then forecasting ahead for a specified number of steps WITHOUT any updates. This evaluates actual forecast skill rather than tracking accuracy. The function returns forecasts, true observations, errors, and distance errors for each lead time.

**`evaluate_sliding_origins()`:**
Implements sliding origin evaluation, where forecasts are generated from multiple points along each storm track rather than only from the storm origin. This provides more robust and representative forecast statistics by sampling diverse storm phases (early development, mature stage, decay, etc.).

For each storm, the function:
- Identifies valid forecast origins (requiring sufficient history and future data)
- Samples origins evenly spaced along the track (configurable number per storm)
- Generates open-loop forecasts from each origin for multiple lead times
- Aggregates errors across all origins and storms

**Open-Loop Forecast Results:**
Evaluation with sliding origins on 20 storms with 555 total forecast instances shows realistic error growth with lead time:

- **6 hours**: Mean error 11.90 km, RMSE 15.86 km
- **12 hours**: Mean error 24.53 km, RMSE 34.50 km  
- **24 hours**: Mean error 58.13 km, RMSE 83.79 km
- **48 hours**: Mean error 158.95 km, RMSE 216.74 km
- **72 hours**: Mean error 286.08 km, RMSE 379.35 km

These results demonstrate the expected monotonic error growth with increasing lead time, confirming that the open-loop forecasting correctly captures forecast skill rather than filtering accuracy. The errors are realistic for hurricane track forecasting, with 48-hour errors comparable to operational model guidance.

#### Visualization and Analysis Tools

**Track Visualization:**
Functions convert metric coordinates back to latitude/longitude for geographic visualization. The track plots display actual storm tracks alongside Kalman filter predictions, enabling visual assessment of filter performance.

**Error Analysis:**
Error plots show forecast errors over time, allowing identification of periods where the filter performs well or poorly. This helps understand model limitations and identify storm phases where predictions are less reliable.

**Spaghetti Plots:**
Monte Carlo simulation generates multiple forecast paths from the current filter state, creating probabilistic "spaghetti" plots that visualize forecast uncertainty. This enables assessment of forecast spread and uncertainty quantification.

#### Key Implementation Fixes

Several critical issues were identified and resolved during implementation:

**Issue 1: Observation Noise Estimation**
The original implementation hard-coded R to an unrealistically small value (0.25 km²), leading to overconfident observations. The fix estimates R from actual observation residuals during filtering, providing realistic uncertainty estimates (approximately 265 km² and 169 km² for x and y coordinates).

**Issue 2: Filtering vs Forecasting Evaluation**  
The original `run_kalman_filter_on_storm()` function updates with true observations at every step, evaluating filtering accuracy rather than forecasting skill. New functions (`open_loop_forecast()`, `evaluate_sliding_origins()`) were implemented to properly evaluate open-loop forecasting where no future observations are used.

**Issue 3: Single Origin Evaluation**
Initial evaluation only forecasted from storm origins, biasing results toward early straight-moving phases. Sliding origin evaluation addresses this by forecasting from multiple points along each track, providing representative statistics across diverse storm phases.

**Issue 4: Feature Extraction Robustness**
Feature extraction was improved to handle pandas Series and array types safely, preventing type errors when accessing feature values from DataFrame rows. A helper function ensures scalar values are extracted correctly for parameter adaptation.

#### Findings and Insights

The Kalman filter implementation demonstrates that a relatively simple constant velocity model can achieve reasonable forecast accuracy for hurricane tracks, particularly at short lead times. The feature-adaptive parameter scaling provides significant flexibility to handle diverse storm conditions.

Key findings include:

1. **Short-term accuracy**: 6-hour forecasts achieve approximately 12 km mean error, suitable for operational use in track prediction.

2. **Error growth pattern**: Errors increase approximately quadratically with lead time, consistent with cumulative process noise in the constant velocity model. This matches expectations for linear dynamical systems.

3. **Feature adaptation value**: Adaptive Q scaling based on storm characteristics enables better handling of turning storms, land interactions, and regime changes, though the constant velocity assumption still limits performance for sharp turns.

4. **Observation uncertainty**: Estimated observation noise is larger than initially assumed, reflecting realistic best-track uncertainty and coordinate conversion errors. This leads to more appropriate uncertainty quantification.

5. **Limitations**: The constant velocity model struggles with rapid motion changes, sharp turns, and extratropical transitions. Future improvements could incorporate acceleration terms or non-linear dynamics.

The implementation provides a solid foundation for probabilistic hurricane track forecasting, with clear paths for enhancement through more sophisticated dynamics models or machine learning approaches.

#### Advanced Improvements: Data-Driven Parameter Estimation and Hyperparameter Tuning

Following the initial implementation, significant improvements were made to enhance the Kalman filter's adaptability and performance through systematic parameter estimation and hyperparameter optimization. These improvements address the limitations of fixed heuristic parameters and enable the model to learn optimal adaptive behavior from training data.

**Process and Observation Noise Estimation from Training Data**

The original implementation used fixed default values for process noise covariance (Q) and observation noise covariance (R), which did not reflect the actual uncertainty characteristics of hurricane motion. A new estimation framework was implemented that directly learns these parameters from training data. The `estimate_process_noise()` function computes Q by analyzing state transition residuals across training storms. For each storm, the function compares actual state transitions with predictions from the constant velocity model, collecting innovation residuals that represent the true process uncertainty. The covariance of these residuals across all training storms provides an empirical estimate of Q that captures the inherent variability in hurricane motion dynamics.

Similarly, observation noise R is estimated using a simplified approach that sets a fixed diagonal covariance based on typical best-track position uncertainty. Best-track data has approximately 0.5 to 1.0 km standard deviation in position measurements, leading to an R matrix with 0.25 to 1.0 km² variance per coordinate. The `fit_QR()` wrapper function coordinates both estimations, providing a unified interface for parameter fitting that ensures Q and R are estimated consistently from the same training subset.

**Time-Varying Adaptive Process Noise**

The original feature-adaptive approach used fixed heuristic scaling factors that were not optimized for actual forecast performance. This was replaced with a more flexible system that computes time-varying process noise Q_t at each forecast step based on storm features. The `compute_Q_scale()` function calculates a scaling factor s_t that multiplies the base Q to produce Q_t = s_t × Q_base. This scaling factor incorporates multiple storm characteristics including track curvature (clipped at the 95th percentile to prevent extreme values), land approach indicators, motion regime classifications, and latitude regime information.

The scaling computation was made robust to handle missing features gracefully. Rather than requiring all features to be present, the system now works with any available subset of features, preventing silent disabling of adaptive behavior when some features are missing. Each feature term is only applied if that feature column exists in the data, and the function safely handles NaN values and missing keys. This lenient approach ensures adaptive Q remains functional even with partial feature sets.

**Hyperparameter Tuning Framework**

A comprehensive hyperparameter tuning system was implemented to optimize the adaptive Q scaling parameters. The tuning framework uses random search over a predefined parameter space, evaluating each candidate set of parameters on a validation subset. The training data is split into three parts: inner training set (80% of training storms) for fitting Q and R, validation set (20% of training storms) for hyperparameter evaluation, and test set (20% of all storms) reserved for final evaluation only.

The `tune_adaptive_Q()` function implements the tuning loop, sampling adaptive parameters from reasonable ranges for each trial. For each sampled parameter set, the function fits Q and R on the inner training set, then evaluates open-loop forecasts on the validation set using those parameters. The objective function is a weighted combination of 24-hour and 48-hour RMSE, providing a balanced metric that emphasizes medium-range forecast accuracy. The tuning process tracks the best parameters across all trials and provides detailed logging of trial results.

To ensure tuning efficiency, the evaluation uses subsets of validation storms and limits the number of forecast origins per storm. The tuning function includes verification checks that sample scaling factors to confirm parameters are actually affecting the model behavior. This prevents situations where tuning appears to run but parameters have no effect due to implementation issues.

**Critical Implementation Fixes**

Several critical logical errors were identified and resolved that were preventing the adaptive Q system from functioning correctly. The most significant issue was a feature gate that required all six feature columns to be present before enabling adaptive Q. This strict requirement meant that if any single feature was missing, adaptive Q would silently disable, causing all tuning efforts to have no effect. The fix changed the requirement to only needing at least one feature column, with the scaling function gracefully handling missing features.

Another critical issue involved parameter shadowing, where the original `adaptive_params` dictionary defined early in the notebook could be used instead of the tuned `best_params_tuned` values. This was resolved by renaming the default parameters to `default_adaptive_params` and ensuring all final evaluations explicitly use `best_params_tuned` with error checking to prevent accidental use of default values.

The forecast pathway was also clarified to distinguish between the old `open_loop_forecast()` function that uses internal heuristic adaptation and the new `open_loop_forecast_improved()` function that uses the tuned parameters. The improved function explicitly accepts Q_base, use_adaptive_Q flag, and adaptive_params as arguments, ensuring tuned parameters are properly propagated through the forecast chain.

**Validation and Verification Systems**

Comprehensive verification systems were added to ensure the improvements are functioning correctly. A feature availability check examines which adaptive Q features are present in the dataset and reports coverage statistics. This helps identify when adaptive Q may be limited by missing features. The tuning process includes verification prints that show sample scaling factors for early trials, confirming that different parameter sets produce different Q scaling behavior.

A verification test compares baseline forecasts (with adaptive Q disabled) against improved forecasts (with adaptive Q enabled) on a single test storm. This test prints debug information showing scaling factors and Q values at each step, making it clear when adaptive Q is actually being applied. The test confirms that forecast errors differ between baseline and improved models, proving that the adaptive system is functioning.

**Results and Impact**

The improvements enable systematic optimization of adaptive Q parameters rather than relying on fixed heuristics. The data-driven Q and R estimation provides more realistic uncertainty quantification that reflects actual hurricane motion variability. The hyperparameter tuning framework allows the model to learn optimal scaling factors from validation performance, potentially improving forecast accuracy beyond what fixed parameters can achieve.

The lenient feature handling ensures adaptive Q remains functional across diverse data conditions, while the verification systems provide confidence that improvements are actually being applied. The separation of training, validation, and test sets ensures proper evaluation without data leakage, and the explicit parameter passing prevents accidental use of default values in final evaluations.

These improvements transform the Kalman filter from a model with fixed adaptive heuristics into a system that can learn optimal adaptive behavior from data, providing a foundation for continued performance enhancement through systematic hyperparameter optimization.

**Feature Validation: Testing the Predictive Power of Adaptive Features**

A critical validation step was implemented to empirically test whether the chosen adaptive features actually predict forecast uncertainty. This analysis addresses a fundamental question: do features like track curvature, land approach, and motion regime correlate with periods where the Kalman filter makes larger prediction errors? The validation framework runs the base non-adaptive model on training storms and collects innovation statistics at each time step. For each forecast step, the system records the innovation v_t = y_t - H*x_pred, computes innovation squared (v_t²) as a measure of prediction error, and associates these errors with the corresponding feature values at that time step.

The `collect_innovation_data()` function implements this data collection by running the Kalman filter with fixed Q and R parameters, then extracting innovations and features at each time step across hundreds of training storms. This creates a dataset linking feature values to actual forecast errors, enabling direct correlation analysis. The function also computes Mahalanobis distance, which provides a more principled uncertainty measure that accounts for the innovation covariance structure.

The `analyze_feature_uncertainty_correlation()` function performs the correlation analysis, testing whether high feature values correspond to larger innovation squared values. For track curvature, the analysis compares innovation squared between high-curvature periods (above median) and low-curvature periods (below median), computing a ratio that indicates how much larger errors are during turning phases. Similarly, the analysis compares innovation squared when storms are approaching land versus when they are over open water, and examines differences across motion regimes and latitude regimes.

The analysis also includes an optional regression-based approach through the `fit_learned_scaling()` function, which attempts to learn optimal scaling factors directly from the innovation data. This function fits a linear regression model in log space: log(s_t²) ≈ β₀ + β₁*curvature + β₂*land + β₃*motion_regime + β₄*latitude_regime. If this regression achieves high R², it suggests the features are predictive and provides learned coefficients that could replace hand-tuned parameters. If R² is low, it indicates the features do not reliably predict uncertainty.

**Validation Results and Implications**

The feature validation analysis revealed that the adaptive features show weak correlation with innovation variance. Track curvature, while showing some relationship with forecast errors, does not demonstrate a strong enough correlation to reliably target periods of increased uncertainty. The ratio between high-curvature and low-curvature innovation squared values is typically close to 1.0, indicating that periods of high curvature do not consistently correspond to larger forecast errors. Similarly, the land approach indicator shows minimal difference in innovation squared between storms near land and those over open water, with ratios typically near unity.

Motion regime and latitude regime classifications also fail to show strong predictive power for forecast uncertainty. The mean innovation squared values across different regimes are similar, suggesting that these categorical features do not capture the conditions where the constant velocity model struggles most. The regression model attempting to learn scaling factors from features achieves low R² scores, typically explaining less than 10% of the variance in innovation squared. This indicates that the chosen features collectively provide limited predictive power for forecast uncertainty.

These findings have important implications for the adaptive Q approach. Since the adaptive features do not reliably predict where forecast errors will be large, inflating Q based on these signals does not effectively target periods of increased storm uncertainty. The adaptive Q system may increase process noise during high-curvature periods or when approaching land, but if these periods do not actually correspond to larger forecast errors, the adaptation provides no benefit and may even degrade performance by over-inflating uncertainty when it is not needed.

As a result of this weak feature-uncertainty correlation, the adaptive-Q model does not significantly outperform the best constant-Q baseline. While the hyperparameter tuning framework successfully identifies parameter sets that work, the fundamental limitation is that the features themselves are not predictive enough to enable meaningful adaptation. This provides a clean explanation for why adaptive Q improvements are modest: the chosen features do not actually line up with where the filter makes big errors. The constant velocity model's prediction errors are driven more by the inherent limitations of the linear dynamics assumption than by the storm characteristics captured in the adaptive features.

This validation step demonstrates the importance of empirically testing feature assumptions rather than assuming that intuitive features will be predictive. The analysis framework provides a principled way to evaluate whether adaptive approaches are justified, and in this case, it reveals that more sophisticated feature engineering or different adaptive strategies may be needed to achieve significant performance gains beyond the constant-Q baseline.

---

## Phase 4: Visualization and Results Analysis (COMPLETED)

### File: `visualizations.ipynb`

The visualization suite provides comprehensive analysis and interpretation of the Kalman filter's performance through a series of carefully designed figures that address different aspects of forecast accuracy, error distribution, and model behavior. These visualizations transform raw numerical results into interpretable insights that demonstrate both the strengths and limitations of the forecasting system.

#### Error Distribution Analysis

The distribution of RMSE across test storms reveals how forecast accuracy varies across the entire test dataset. This histogram shows how many storms achieved each RMSE value, providing a big-picture view of overall forecasting system performance. The distribution exhibits a strong right-skewed pattern where most storms cluster around low RMSE values between 15 and 35 kilometers, demonstrating that the Kalman filter performs very well on the majority of storms. The long tail extending to the right indicates the existence of storms that were inherently much harder to predict, typically those with irregular motion patterns, short lifetimes, or sudden direction changes. The mean and median RMSE are marked with dashed lines, and since the median sits slightly lower than the mean, this confirms that a small number of large-error storms are raising the average. This visualization answers the fundamental question of how good the forecasting system is overall, and it clearly illustrates that for most storms, the Kalman filter produces accurate short-term forecasts.

The distribution of mean error focuses not on total forecast error but on whether the Kalman filter tends to consistently over-predict or under-predict storm position. A forecasting system with strong directional bias would consistently push predicted locations in one direction, which would be problematic for operational use. What we observe is that the distribution of mean error per storm is clustered tightly around relatively small values, with most storms having mean errors between 10 and 25 kilometers. The overall mean bias is approximately 21 kilometers, which represents absolute bias rather than directional bias, simply indicating the typical magnitude of average forecast misplacement. The right-skewed shape indicates that most storms have small systematic error while a few storms have larger consistent forecast offsets. This tells us that the Kalman filter is mostly unbiased and does not drift incorrectly in a systematic direction, which is critical for trustworthiness in operational forecasting applications.

The scatter plot examining RMSE versus storm length investigates whether storm complexity, measured by the number of observations or track length, influences forecast accuracy. Each point represents one storm, plotted by how many data points it had and what its RMSE was, with color intensity representing the magnitude of error. The visualization reveals that storms with very few observations are much more widely scattered in RMSE, showing high variability, noise, and many high-error outliers. As storms become longer-lived and accumulate more observations, particularly over 50, 100, and 150 points, forecast errors become more tightly clustered around lower values. In other words, storms that live long enough for the Kalman filter to observe more motion patterns become easier to predict. This makes intuitive sense because the Kalman filter relies heavily on learning the storm's velocity and directional trends, and longer tracks simply provide more information for stable estimation. The gradual downward trend as storm length increases confirms this relationship and validates that the filter benefits from having more temporal context.

The cumulative distribution of RMSE answers a foundational question about forecast reliability: how often is the Kalman filter accurate? Instead of showing how many storms fall into each error bin, this plot shows the percentage of storms that achieve an error below any given threshold. By reading the curve, we can see that approximately 50 percent of storms have an RMSE below about 23 kilometers, which represents the median, and 90 percent of storms fall below roughly 43 kilometers. This visualization is extremely useful because it communicates forecast reliability in intuitive terms, telling us not just how accurate the system is on average, but how frequently we can expect it to achieve certain accuracy levels. In a real forecast setting, this is the plot a meteorologist or risk analyst would care about because it describes probabilistic reliability. The flat plateau at the end shows that very few storms exceed high RMSE values, again confirming that only rare storms are truly problematic for the filter.

The boxplot visualization grouping storms by track length categories reinforces the relationship seen in the scatter plot. Short storms with fewer than 10 observations have wide error variability, with higher medians and many extreme outliers, meaning forecasts for these storms are harder and less stable. As storm length increases through the categories of 10-20, 20-30, 30-50, 50-100, and over 100 observations, the median RMSE decreases, the interquartile ranges shrink, and the outliers become fewer and less extreme. In other words, longer-lived storms produce more predictable behavior, and the Kalman filter benefits from having more time steps to accurately track velocity and curvature. This plot helps justify the robustness of the model by showing that, although some storm categories are inherently difficult, the filter behaves very consistently when provided adequate data.

#### Error Distribution by Lead Time with Normal Overlay

The error distribution plots at different lead times provide a detailed examination of how forecast error distributions change as the forecast horizon extends. Each plot compares three different representations of the same forecast-error distribution: a histogram that gives a raw count-based view of the errors, a KDE curve that smooths the empirical distribution into a continuous shape, and a normal distribution curve generated using the sample mean and standard deviation. At 6 hours, the KDE curve peaks sharply at low errors, confirming that most forecasts lie under about 15 kilometers, and the median is lower than the mean because a handful of larger errors create a right-skewed tail. The normal curve is wider and flatter than the empirical KDE, which shows that a Gaussian assumption exaggerates the probability of large errors and understates the clustering near zero.

The 12-hour plot repeats the same comparison, but now the entire distribution has shifted to the right. Errors have nearly doubled, with the mean increasing to around 25 kilometers. The KDE still shows a strong right-skew, and the tail becomes heavier, indicating that a few storms produce unusually large errors early in the forecast horizon. The normal curve again overestimates the thickness of this tail. What changes most clearly here is the growing divergence between the empirical distribution and the symmetric Gaussian shape, which reinforces that hurricane forecast errors do not follow a simple normal distribution and instead exhibit asymmetry, skew, and heavy tails.

At 24 hours, the distribution becomes wider and significantly more skewed. The mean exceeds 58 kilometers and the spread more than doubles. The KDE curve shows a long, gradual decline, reflecting that many storms are forecast fairly well but a substantial number incur large deviations that stretch beyond 200 kilometers. The normal curve, although centered correctly by design, again lacks the sharp peak and exaggerates the symmetry, showing that the Gaussian model fails to capture the empirical pattern of many small errors and a minority of extreme failures.

The 48-hour distribution intensifies this pattern, with errors climbing into the hundreds of kilometers. The KDE still shows a strong right skew, but the spread widens enough that the histogram bars thin out because each bin is now covering a much larger continuous range. Forecast variance increases dramatically and the tail becomes very long. At this point, the mismatch between the empirical KDE and the normal curve becomes even clearer because the Gaussian curve does not capture the massive spread and asymmetry of the observed errors.

By the 72-hour horizon, the distribution becomes extremely dispersed, with a mean above 280 kilometers and individual forecasts missing the true storm position by more than 1400 kilometers. The KDE curve is very broad and the histogram only captures the rough structure of the distribution because the sample is now stretched so widely. This figure captures the increasing difficulty of forecasting storm motion multiple days into the future, and highlights that uncertainty grows nonlinearly as lead time increases.

#### Cumulative Distribution of Forecast Error

The cumulative distribution plot complements the individual error distributions by showing the fraction of forecasts that fall below any given error threshold. This cumulative perspective illustrates how rapidly performance deteriorates with lead time. At 24 hours, half of all forecasts fall within about 34 kilometers, at 48 hours the same percentile shifts to around 111 kilometers, and at 72 hours the median climbs to roughly 190 kilometers. The curves rise steeply at short lead times but flatten as errors exceed several hundred kilometers, reflecting that only a small number of forecasts ever reach extremely large error levels. This provides a clear summary of reliability: early horizons are tightly controlled, while long-range predictions become unreliable and highly variable. The plot enables direct interpretation of forecast reliability in probabilistic terms, which is essential for practical applications such as constructing forecast cones or uncertainty bounds.

#### Innovation Analysis

The innovation components plot shows the x- and y-components of the innovation vector, which is the difference between the observed storm position and the model's predicted observation. In a well-behaved Kalman filter, these innovations should cluster around zero with no systematic directional bias, and the shape of the point cloud should resemble a roughly circular scatter. The plot meets these expectations. The points are centered near the origin, meaning the model does not systematically overshoot or undershoot in either longitude or latitude. The fact that the scatter is reasonably symmetric indicates that the variance is similar in both coordinates, and the absence of a stretched elliptical pattern suggests that the two components are not strongly correlated. A few outliers extending to large distances represent a small number of sudden storm deviations, which is normal in hurricane data. Overall, the innovations appear consistent with the linear-Gaussian assumptions that the Kalman filter relies on.

The innovation analysis also examines how prediction uncertainty relates to storm features such as track curvature, land proximity, motion regime, and latitude regime. The plots comparing innovation squared across different feature categories reveal that these features show weak correlation with forecast uncertainty. Track curvature, while showing some relationship with forecast errors, does not demonstrate a strong enough correlation to reliably target periods of increased uncertainty. The ratio between high-curvature and low-curvature innovation squared values is typically close to 1.0, indicating that periods of high curvature do not consistently correspond to larger forecast errors. Similarly, the land approach indicator shows minimal difference in innovation squared between storms near land and those over open water. Motion regime and latitude regime classifications also fail to show strong predictive power for forecast uncertainty, with mean innovation squared values across different regimes being similar. These findings explain why the adaptive-Q model does not significantly outperform the baseline constant-Q model, as the chosen features do not actually line up with where the filter makes big errors.

#### Model Comparison: Baseline vs Adaptive-Q

The RMSE comparison between the baseline Kalman filter and the adaptive-Q version demonstrates how forecast accuracy changes with lead time. Both models naturally exhibit increasing error as the forecast extends farther out, but the adaptive-Q filter performs slightly better at shorter horizons. At 6, 12, and 24 hours, the adaptive-Q model produces marginally smaller RMSE values, indicating that adjusting the process noise in response to environmental cues improves short-term track prediction. However, as the lead time approaches 48 and 72 hours, the advantage disappears. Once the forecast horizon becomes long enough for uncertainty to accumulate, the benefits of adapting Q are overwhelmed by the natural growth of meteorological variability. This behavior is consistent with the literature, where short-term improvements are possible but long-range tracks remain difficult for any linear filter to refine.

The error distribution boxplots for the two models reinforce this interpretation. For each lead time, the median forecast error of the baseline and adaptive-Q filters is almost identical, meaning neither model introduces systematic bias. The primary difference occurs in the spread of the middle portion of the errors at shorter lead times, where the adaptive-Q method slightly narrows the interquartile range. This indicates that the adaptive-Q filter not only reduces average error but also stabilizes the predictions by reducing variability for a substantial portion of the storms. By the time the lead time reaches forty-eight and seventy-two hours, the distributions overlap almost completely. This shows that the additional adaptivity is no longer influential when uncertainty has compounded.

The improvement-by-lead-time figure shows the same pattern from another angle. The percent improvement is strongest at very short lead times, then declines steadily. The improvement at six hours is close to two percent, and it remains above one percent at twelve and twenty-four hours. However, by forty-eight hours the improvement is almost zero, and by seventy-two hours the adaptive-Q version actually performs slightly worse, although the difference is so small that it is essentially within noise. This confirms that the adaptive-Q filter provides a measurable but modest benefit at short time scales, and that long-range forecasts remain largely dependent on the inherent limits of hurricane predictability.

The cumulative error distribution plot further supports this conclusion. At twenty-four hours, the adaptive-Q curve lies slightly to the left of the baseline curve, which means that for nearly every percentile, the adaptive-Q filter produces errors that are marginally smaller. This is precisely what a well-behaved improvement should look like. At forty-eight hours, the two curves nearly overlap, which again demonstrates that long-term uncertainty reduces the usefulness of adaptively scaling Q.

The scatter plot comparing adaptive-Q errors to baseline errors offers a point-by-point view of how individual forecasts change. Since the diagonal line represents no difference between the models, points below the line indicate situations where adaptive-Q helped. The vast majority of points lie very close to the diagonal, showing that the two models behave almost identically. A smaller set of points fall just below the diagonal, which reflects the small improvements observed in the summary statistics. This plot makes clear that the advantage of the adaptive-Q method is real but subtle and does not transform the error structure dramatically.

#### Spaghetti Plots: Error Trajectories and Track Visualization

The error spaghetti plot shows forecast position error over time for many storms, with one line per storm representing its error trajectory across different lead times. Each line traces how error grows for an individual storm as the forecast extends from 6 to 72 hours. The plot includes mean and median error trajectories as reference lines, along with an interquartile range shading that shows the spread of errors across storms. This visualization reveals that most storms follow similar error growth patterns, with errors increasing approximately quadratically with lead time, consistent with cumulative process noise in the constant velocity model. However, some storms show much steeper error growth, indicating periods where the linear motion assumption breaks down. The plot helps identify which storms contribute to heavy-tail outliers and where the Kalman filter breaks down, typically around 36 to 48 hours when uncertainty has compounded sufficiently.

The trajectory spaghetti plot shows true tracks for many storms aligned at a shared forecast origin, allowing their shapes and directional tendencies to be visually compared. Because each storm begins from the same reference point, differences in curvature, forward speed, and turning behavior are easy to see. This plot illustrates the diversity of hurricane trajectories the Kalman filter is expected to track, from slow recurving storms to fast, straight-moving ones. The increasing uncertainty in the error-distribution plots reflects the genuine diversity of storm motion seen here. When forecast tracks are available, this plot would show predicted versus real trajectories, enabling visual assessment of systematic biases such as too much westward drift or consistent under-prediction of recurvature.

The forecast ensemble fan plot focuses on a single storm and shows how forecast uncertainty grows within that storm by displaying forecasts from multiple origins along the track. The true track is shown as a single black line, while forecast origins are marked at various points. When forecast positions are available, this would create a fan or cone-of-uncertainty style visualization where multiple predicted tracks fan out from different origins, demonstrating how uncertainty expands with time. This plot is particularly powerful because it visually demonstrates the core weakness of linear filtering: long-range predictions blow up as uncertainty accumulates. It also shows whether the Kalman filter stays on track early but diverges after 48 hours, and whether certain points such as recurvature cause catastrophic forecast spread.

#### Overall Interpretation

Together, these figures provide a comprehensive understanding of how the Kalman filter behaves across thousands of hurricane tracks. The RMSE distribution tells us the absolute forecasting accuracy, while the bias distribution tells us whether the filter systematically misplaces storms, which it does not. The relationship with storm length reveals an important structural pattern: more observations enable more accurate velocity estimation, and therefore better position predictions. The cumulative RMSE curve provides an intuitive summary of reliability, which is essential for practical applications such as constructing forecast cones or uncertainty bounds. The error distribution plots by lead time demonstrate that forecast errors do not follow a normal distribution but instead exhibit heavy tails and right-skew, which has implications for uncertainty quantification. The innovation analysis validates that the filter behaves consistently with linear-Gaussian assumptions, while the model comparison plots show that adaptive-Q provides modest improvements at short lead times but cannot overcome the fundamental limitations of linear dynamics at long horizons. The spaghetti plots provide both aggregate error trajectory analysis and individual storm track visualization, completing the picture of forecast performance across different scales of analysis.

These visualizations collectively validate that the model is performing well and behaving consistently across storm types, while also clearly identifying the limitations of the constant velocity assumption for long-range forecasting. The comprehensive visualization suite enables both technical evaluation of model performance and intuitive communication of forecast reliability to end users who need to make decisions based on hurricane track predictions.

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
Kalman Filter Implementation (Kalman_Filter.ipynb) [COMPLETED]
    ├── Model design with metric coordinates
    ├── Parameter estimation from training data
    ├── Filter implementation with feature adaptation
    ├── Open-loop forecasting evaluation
    └── Sliding origin evaluation
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

## Next Steps (Future Enhancements)

1. **Model Improvements**
   - Incorporate acceleration terms into state vector for better handling of motion changes
   - Implement non-linear dynamics models for improved accuracy during sharp turns
   - Add intensity forecasting using wind speed or pressure observations

2. **Advanced Features**
   - Ensemble forecasting with multiple model configurations
   - Machine learning enhancements for parameter adaptation
   - Basin-specific model tuning and evaluation

3. **Evaluation Enhancements**
   - Compare with operational forecast models (CLIPER, SHIFOR, NHC guidance)
   - Detailed analysis of error patterns by storm characteristics
   - Cross-validation and robustness testing across different time periods

4. **Visualization and Reporting**
   - Interactive forecast visualization tools
   - Automated forecast report generation
   - Real-time forecast monitoring dashboard

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
- Project Proposal: `Report/CSE150A Report.pdf`

