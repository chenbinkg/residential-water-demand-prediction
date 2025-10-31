# Water Demand Prediction

This repository contains a comprehensive framework for predicting residential water demand in the Wellington region, New Zealand. It combines traditional SageMaker Autopilot workflows with an advanced experimental framework featuring:

- **Statistical Analysis**: ACF/PACF analysis with stationarity testing
- **Feature Engineering**: Systematic feature ablation with multi-algorithm consensus
- **Advanced Modeling**: 3-layer stacked ensemble (51 models per site) inspired by AutoGluon
- **Visualization**: Heatmaps, time series plots, and performance comparisons
- **Production Ready**: Trained models saved for inference with comprehensive evaluation

## Repository Structure

```
.
├── data/                                    # Training and reference data
│   ├── cm.csv                              # Monthly coefficients
│   ├── {site} xpARA.csv                    # Site-specific coefficients
│   └── processed/                          # Processed data with lag features
├── scripts/
│   ├── feature_analysis.py                 # Core analysis framework
│   ├── add_lag_features.py                 # Lag feature engineering
│   ├── prep_inference_data.py              # Inference data preparation
│   ├── prep_simulation_data.py             # Simulation data preparation
│   └── prep_*_results.py                   # Results processing
├── results/                                 # Experimental outputs
│   ├── acf/                                # ACF/PACF analysis
│   ├── models/                             # Trained ensemble models
│   ├── ablation_{site}.csv                 # Feature ablation results
│   ├── algorithms_{site}.csv               # Algorithm benchmarks
│   ├── importance_{site}.csv               # Feature importance
│   └── ensemble_metrics_{site}.csv         # Ensemble performance
├── run_experiments.py                       # Main experiment runner
├── EXPERIMENTS_README.md                    # Detailed experimental documentation
├── requirements.txt                         # Python dependencies
├── autopilot_*.ipynb                        # SageMaker Autopilot notebooks
└── consolidate_*.ipynb                      # Results consolidation notebooks
```


## Features

### Experimental Framework

1. **ACF/PACF Analysis**
   - Stationarity testing (ADF, KPSS, SigLag)
   - Autocorrelation and partial autocorrelation analysis
   - Lag feature recommendations based on statistical significance
   - Time series visualization (original vs differenced)

2. **Feature Ablation Study**
   - Multi-algorithm consensus approach (7 algorithms)
   - Cumulative feature group evaluation (28 groups)
   - Identifies effective features (>50% algorithm agreement)
   - Excludes target lag features from final models

3. **Algorithm Benchmarking**
   - Compares 7 algorithms: XGBoost, GradientBoosting, CatBoost, RandomForest, ExtraTrees, MLP, LinearRegression
   - Uses consensus features from ablation study
   - Generates performance heatmaps across sites

4. **Feature Importance Analysis**
   - RandomForest-based importance rankings
   - Cross-site comparison heatmaps
   - Top feature identification

5. **Stacked Ensemble Modeling**
   - 3-layer architecture inspired by AutoGluon
   - Layer 1: 7 base models with 5-fold bagging (35 models)
   - Layer 2: 3 stacked models with 5-fold bagging (15 models)
   - Layer 3: Greedy weighted ensemble (1 meta-model)
   - Total: 51 models per site
   - Saves trained models for inference
   - Generates time series and scatter plot visualizations

### SageMaker Autopilot Workflows

- **Training**: `autopilot_demand_training.ipynb`
- **Inference**: `autopilot_demand_inference.ipynb`
- **Simulation**: Instance-specific notebooks (c5m5xlarge, m4c4, m5large_c4c5-2x)
- **Consolidation**: Results aggregation notebooks
- **Exploration**: Data exploration and candidate definition notebooks

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run all experiments (ACF, ablation, benchmark, importance)
python run_experiments.py --all

# Run specific experiments
python run_experiments.py --acf              # ACF/PACF analysis only
python run_experiments.py --ablation         # Feature ablation only
python run_experiments.py --benchmark        # Algorithm benchmark only
python run_experiments.py --importance       # Feature importance only
python run_experiments.py --ensemble         # Stacked ensemble training (expensive)

# Combine multiple experiments
python run_experiments.py --ablation --benchmark --ensemble
```

### Generate Visualizations

```bash
# Algorithm performance heatmap
python -c "from scripts.feature_analysis import plot_algorithm_benchmark_heatmap; plot_algorithm_benchmark_heatmap()"

# Feature importance heatmap (top 20 features)
python -c "from scripts.feature_analysis import plot_feature_importance_heatmap; plot_feature_importance_heatmap(top_n=20)"
```

### Using Trained Ensemble Models

```python
import pickle

# Load trained ensemble model
with open('results/models/ensemble_Lower Hutt.pkl', 'rb') as f:
    model = pickle.load(f)

# Access model components
layer1_models = model['layer1_models']
layer2_models = model['layer2_models']
weights = model['ensemble_weights']
features = model['feature_names']
metrics = model['metrics']
```

### SageMaker Autopilot Usage

1. **Training**: Use `autopilot_demand_training.ipynb` to train models
2. **Inference**: Use `autopilot_demand_inference.ipynb` for predictions
3. **Simulation**: Use instance-specific notebooks for simulations
4. **Consolidation**: Use consolidation notebooks to aggregate results

## Requirements

- Python 3.8+
- Jupyter Notebook
- Amazon SageMaker (for Autopilot workflows)
- AWS credentials configured (for S3 access)

### Key Dependencies

- **Core**: pandas, numpy, scikit-learn
- **ML**: xgboost, catboost
- **Stats**: statsmodels
- **Visualization**: matplotlib, seaborn
- **AWS**: boto3, sagemaker

See `requirements.txt` for complete list.

## Target Sites

The framework analyzes 10 water supply zones in the Wellington region:

1. Lower Hutt
2. Wellington High Moa
3. North Wellington Porirua
4. Petone
5. Porirua
6. Upper Hutt
7. Wainuiomata
8. North Wellington Moa
9. Wellington High Western
10. Wellington Low Level

## Key Findings

### Feature Importance
Top features across sites:
- PE (Potential Evapotranspiration)
- API (Antecedent Precipitation Index)
- Sun (Sunshine hours)
- doy (Day of year)
- ANcyc (Annual cycle)
- Tmax (Maximum temperature)

### Model Performance
- Tree-based models (XGBoost, RandomForest, CatBoost) consistently outperform linear models
- Ensemble models achieve R² > 0.85 on most sites
- Stacked ensemble provides 2-5% improvement over single models

### Known Issues

**Temporal Bias (2021-2024 Winter Underprediction)**

The model systematically underpredicts water demand during recent winters (2021-2024) across multiple sites. This is likely due to:

1. **Regime change**: COVID-19 and behavioral shifts not captured by current features
2. **Missing temporal features**: No year trend or long-term indicators
3. **Model limitation**: Tree-based models cannot extrapolate beyond training range
4. **Short-term lags only**: Current lag features only look back 1-4 days

**Recommended solutions**:
- Add temporal trend features (year, years_since_2020, is_covid_era)
- Add year-over-year lag features (demand_lag_365, demand_change_yoy)
- Add long-term rolling statistics (demand_rolling_365)
- Use time-based split for validation
- Consider external data (COVID indicators, water pricing, conservation campaigns)

See `EXPERIMENTS_README.md` for detailed analysis and recommendations.

## Documentation

- **EXPERIMENTS_README.md**: Comprehensive experimental framework documentation
- **requirements.txt**: Python package dependencies
- **Notebooks**: Individual notebook documentation within each file

## Output Files

### Analysis Results
- `results/ablation_{site}.csv` - Feature ablation with consensus
- `results/algorithms_{site}.csv` - Algorithm benchmark results
- `results/importance_{site}.csv` - Feature importance rankings
- `results/ensemble_metrics_{site}.csv` - Ensemble performance

### Visualizations
- `results/acf/{site}/*.png` - ACF/PACF and time series plots
- `results/algorithm_benchmark_heatmap.png` - Cross-site algorithm comparison
- `results/feature_importance_heatmap.png` - Cross-site feature importance
- `results/models/timeseries_{site}.png` - Ensemble predictions over time
- `results/models/scatter_{site}.png` - Actual vs predicted scatter plots

### Models
- `results/models/ensemble_{site}.pkl` - Trained ensemble models (51 models per site)
- `results/models/predictions_{site}.csv` - Predictions with ground truth

## Contributing

For questions or contributions, please refer to the experimental documentation in `EXPERIMENTS_README.md`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NIWA (National Institute of Water and Atmospheric Research)
- Wellington Water
- AWS for SageMaker Autopilot capabilities
- Open-source community for ML libraries