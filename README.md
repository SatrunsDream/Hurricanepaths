# Hurricane Track Prediction using Kalman Filter

This project implements a probabilistic state-space model for hurricane track forecasting using the Kalman filter. The implementation uses historical hurricane track data from IBTrACS (International Best Track Archive for Climate Stewardship) to train and evaluate a constant-velocity Kalman filter with adaptive process noise scaling.

## Project Overview

The Kalman filter provides a framework for sequential Bayesian inference, enabling probabilistic forecasting of hurricane tracks. The model uses a constant velocity state-space representation with position and velocity components, and incorporates feature-adaptive process noise that adjusts based on storm characteristics such as track curvature, land proximity, and motion regimes.

## File Structure

### Analysis Notebooks

- **`eda_cleaning.ipynb`**: Exploratory data analysis and data cleaning. This notebook loads the IBTrACS dataset, performs quality assessment, validates temporal structure, and generates a processed dataset ready for feature engineering.

- **`features_engineering.ipynb`**: Feature engineering for Kalman filter implementation. Transforms cleaned data into a format suitable for state-space modeling, including velocity computation, coordinate conversion, and extraction of advanced features such as track curvature, motion regimes, and landfall proximity.

- **`Kalman_Filter.ipynb`**: Core Kalman filter implementation and evaluation. Contains the state-space model design, parameter estimation from training data, filtering and forecasting functions, and comprehensive evaluation using sliding origin methodology. Includes both baseline and adaptive-Q filter variants.

- **`visualizations.ipynb`**: Comprehensive visualization suite for results analysis. Generates plots for error distributions, forecast accuracy by lead time, innovation analysis, model comparisons, and example storm trajectories. All figures are saved separately in the `figures/` directory.

### Data Files

The `data/` directory contains:

- **`ibtracs.ALL.list.v04r01.csv`**: Raw IBTrACS dataset with hurricane track data from 1842-2025
- **`hurricane_paths_processed.pkl`**: Processed dataset ready for Kalman filter (721,960 observations, 13,450 storms)
- **`kalman_results_*.pkl`**: Kalman filter evaluation results including sliding window forecasts, test set results, and innovation statistics
- **`dataset_summary.json`**: Structured metadata summary of the dataset

### Output Files

The `figures/` directory contains all generated visualizations organized by figure number and type:
- Error distribution plots (fig8, fig9, fig12)
- Innovation analysis plots (fig10)
- Model comparison plots (fig11)
- Cumulative distribution plots (fig13)
- Example trajectory plots (fig14)
- Spaghetti plots (fig15, fig16)
- Ensemble fan plots (fig17)

### Documentation

- **`README.md`**: This file - project overview and file structure
- **`Report/`**: Project reports and methodology documentation
- **`References/`**: Reference papers and documentation

## Key Results

The Kalman filter achieves reasonable forecast accuracy for hurricane tracks, particularly at short lead times:
- **6 hours**: Mean error ~12 km, RMSE ~16 km
- **24 hours**: Mean error ~58 km, RMSE ~84 km
- **48 hours**: Mean error ~159 km, RMSE ~217 km
- **72 hours**: Mean error ~286 km, RMSE ~379 km

The adaptive-Q variant provides modest improvements at short lead times (1-2% RMSE reduction) but the advantage diminishes at longer horizons where uncertainty accumulates.

## Usage

1. Run `eda_cleaning.ipynb` to load and clean the IBTrACS dataset
2. Run `features_engineering.ipynb` to generate processed features
3. Run `Kalman_Filter.ipynb` to train and evaluate the filter
4. Run `visualizations.ipynb` to generate all analysis plots

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn (for some evaluation metrics)