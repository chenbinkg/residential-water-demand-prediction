# Experimental Framework for Technical Paper

## Overview
This framework provides systematic experiments to analyze feature effectiveness and model performance for residential water demand prediction, including advanced stacked ensemble modeling inspired by AutoGluon.

## Experimental Design

### 1. ACF/PACF Analysis
**Purpose**: Validate lag feature selection and assess stationarity

**Variables analyzed**:
- Target: Water demand for each site
- Predictors: Tmax, Sun, PE, Aquifer, API, storage

**Tests performed**:
- ADF (Augmented Dickey-Fuller): Tests for unit root
- KPSS: Tests for stationarity
- SigLag: Counts significant ACF lags (practical test)

**Output**: 
- ACF/PACF plots (original vs first differenced series)
- Time series comparison plots
- Stationarity test results (CSV)
- Lag recommendations based on PACF

### 2. Feature Ablation Study
**Purpose**: Quantify contribution of each feature group with multi-algorithm consensus

**Feature groups** (cumulative):
1. **date_basic**: doy, month, mday, wday
2. **season**: Season
3. **holiday**: is_holiday
4. **tmax**: Tmax
5. **tmax_lag1**: Tmaxlag1
6. **tmax_lag2_3**: Tmaxlag2, Tmaxlag3
7. **sun**: Sun
8. **sun_lag1**: Sunlag1
9. **sun_lag2_3**: Sunlag2, Sunlag3
10. **pe**: PE
11. **pe_lag1_2_3**: PElag1, PElag2, PElag3
12. **rain**: Aquifer, WVRain, KerburnRain
13. **rain_lag1_2_3**: Rainlag1, Rainlag2, Rainlag3
14. **rain_rolling**: Rain_L7DAYS to Rain_L2DAYS
15. **cyclical**: ANcyc
16. **api**: API
17. **api_lag1**: lagAPI
18. **api_lag2_4**: lag2API, lag4API
19. **storage**: storage
20. **storage_lag1**: Storagelag1
21. **storage_lag2_4**: Storagelag2, Storagelag4
22. **restriction**: Restriction level
23. **ftemp**: fTemp
24. **fprecp**: fPrecp
25. **cm**: cm
26. **stat**: stat
27. **stat_lag1_2_3**: statlag1, statlag2, statlag3
28. **target_lag1_2_3**: Targetlag1, Targetlag2, Targetlag3 (excluded from final models)

**Consensus approach**:
- Tests 7 algorithms: RF, GB, CB, XGB, ET, MLP, LR
- Feature group marked effective if >50% algorithms show improvement
- Improvement = both RMSE and MAE decrease

**Metrics**: RMSE, MAE, R²

### 3. Algorithm Benchmarking
**Purpose**: Compare base algorithms on consensus features

**Algorithms**:
- Boosting: XGBoost, GradientBoosting, CatBoost
- Tree-based: RandomForest, ExtraTrees
- Neural Network: MLP
- Linear: LinearRegression

**Configuration**: 
- Uses only consensus features from ablation study
- Excludes target lag features (Targetlag1-3)
- Consistent hyperparameters across sites

### 4. Feature Importance Analysis
**Purpose**: Identify most influential features

**Method**: RandomForest feature importance (200 trees)
**Features**: Only consensus features (excluding target lags)

### 5. Stacked Ensemble Training (NEW)
**Purpose**: Train production-ready ensemble model following AutoGluon architecture

**Architecture**:
- **Layer 1 (Base)**: 7 models (XGB, GB, CB, RF, ET, MLP, LR) with 5-fold bagging
  - Each model trained on K folds → 7 × 5 = 35 models
  - Generates out-of-fold (OOF) predictions
- **Layer 2 (Stack)**: 3 models (RF, XGB, MLP) with 5-fold bagging
  - Uses original features + Layer 1 OOF predictions
  - Each model trained on K folds → 3 × 5 = 15 models
- **Layer 3 (Meta)**: Greedy weighted ensemble
  - Learns optimal weights to combine Layer 2 predictions
  - 1 meta-model
- **Total**: 51 models per site

**Key techniques**:
- Bagging with cross-validation (prevents overfitting)
- Out-of-fold predictions (prevents data leakage)
- Stacked ensembling (combines model strengths)
- Weighted ensemble (optimal model combination)

**Output**:
- Trained ensemble model (PKL)
- Predictions with ground truth (CSV)
- Performance metrics (CSV)
- Time series visualization (PNG)
- Scatter plot with fitted line (PNG)

## Data Split Strategy

**Stratified split** (default):
- Training: 70%
- Testing: 30%
- Stratified by Restriction level to ensure balanced representation
- Randomly distributes data across train/test (not temporal)

**Alternative: Time-based split**:
- Use `split_method='time'` for temporal validation
- First 70% chronologically for training, last 30% for testing
- Ensures model tested on truly unseen future data
- **Recommended for production** to assess temporal generalization

**Note**: Current analysis uses stratified split, which explains why the model sees some 2021-2024 data during training but still underpredicts recent winters. This suggests a structural change in consumption patterns that features cannot capture.

## Usage

### Run all experiments (excluding ensemble):
```bash
python run_experiments.py --all
```

### Run specific experiments:
```bash
# ACF analysis only
python run_experiments.py --acf

# Feature ablation only
python run_experiments.py --ablation

# Algorithm benchmark only
python run_experiments.py --benchmark

# Feature importance only
python run_experiments.py --importance

# Stacked ensemble training (computationally expensive)
python run_experiments.py --ensemble

# Combine multiple
python run_experiments.py --ablation --benchmark --ensemble
```

### Generate visualizations:
```bash
# Algorithm performance heatmap
python -c "from scripts.feature_analysis import plot_algorithm_benchmark_heatmap; plot_algorithm_benchmark_heatmap()"

# Feature importance heatmap
python -c "from scripts.feature_analysis import plot_feature_importance_heatmap; plot_feature_importance_heatmap(top_n=20)"
```

## Output Structure

```
results/
├── acf/
│   └── {site}/
│       ├── {variable}_acf_pacf.png          # ACF/PACF plots (differenced)
│       ├── {variable}_series_comparison.png # Original vs differenced series
│       ├── stationarity_tests.csv           # ADF, KPSS, SigLag results
│       └── lag_recommendations.csv          # PACF-based lag suggestions
├── ablation_{site}.csv                      # Feature ablation with consensus
├── algorithms_{site}.csv                    # Algorithm benchmark results
├── importance_{site}.csv                    # Feature importance rankings
├── ensemble_metrics_{site}.csv              # Ensemble performance metrics
├── algorithm_benchmark_heatmap.png          # R² heatmap across sites
├── feature_importance_heatmap.png           # Top features heatmap
└── models/
    ├── ensemble_{site}.pkl                  # Trained ensemble model
    ├── predictions_{site}.csv               # Predictions + ground truth
    ├── timeseries_{site}.png                # Time series plot (train/test)
    └── scatter_{site}.png                   # Scatter plot with fitted line
```

## Key Findings to Report

1. **ACF Analysis**:
   - Stationarity assessment (ADF, KPSS, SigLag tests)
   - First differencing achieves stationarity for most variables
   - Significant PACF lags justify 1-3 day lag features
   - Lag recommendations based on statistical significance

2. **Feature Ablation**:
   - Multi-algorithm consensus on feature effectiveness
   - RMSE/R²/MAE progression across feature groups
   - Consensus features identified (>50% algorithm agreement)
   - Target lag features excluded from final models

3. **Algorithm Performance**:
   - Best performing algorithm per site
   - Performance variance across algorithms
   - R² heatmap showing consistency across sites
   - Tree-based models generally outperform linear models

4. **Feature Importance**:
   - Top features: PE, API, Sun, doy, ANcyc, Tmax
   - Consistency across sites (heatmap visualization)
   - Weather variables dominate importance rankings

5. **Ensemble Performance**:
   - 51 models per site (35 base + 15 stacked + 1 meta)
   - Weighted ensemble combines model strengths
   - Time series plots show prediction quality
   - Scatter plots reveal systematic biases (e.g., 2021-2024 winter underprediction)

## Recommendations for Paper

### Tables:
1. Stationarity test results (ADF, KPSS, SigLag for key variables)
2. Feature ablation with consensus (showing n_effective and consensus columns)
3. Algorithm comparison (R², RMSE, MAE for all sites)
4. Top 20 features by importance (averaged across sites)
5. Ensemble performance metrics (train/test R², RMSE, MAE)

### Figures:
1. ACF/PACF plots (2-3 key variables showing differenced series)
2. Time series comparison (original vs differenced for stationarity)
3. Algorithm performance heatmap (R² across sites and algorithms)
4. Feature importance heatmap (top 20 features across sites)
5. Ensemble time series predictions (train/test with actual vs predicted)
6. Ensemble scatter plots (actual vs predicted with fitted line)
7. Feature ablation progression (cumulative R² improvement)

## Known Issues & Future Work

### Temporal Bias (2021-2024 Winter Underprediction)
**Observation**: Model systematically underpredicts water demand during 2021-2024 winters across multiple sites (Wainuiomata, Upper Hutt, Wellington Low Level, Lower Hutt, North Wellington Moa, Petone, Porirua).

**Likely causes**:
1. **Regime change**: COVID-19 and behavioral shifts not captured by features
2. **Missing features**: No year trend, COVID indicators, or economic factors
3. **Model limitation**: Tree-based models cannot extrapolate beyond training range
4. **Feature engineering**: Lag features only look back 1-4 days, missing year-over-year changes

**Recommended solutions**:
1. Add temporal trend features: `year`, `years_since_2020`, `is_covid_era`
2. Add year-over-year lags: `demand_lag_365`, `demand_change_yoy`
3. Add long-term rolling statistics: `demand_rolling_365`, `demand_trend`
4. Use time-based split for validation (not stratified)
5. Consider external data: COVID lockdowns, water pricing, conservation campaigns

### Other Considerations
- Focus on 3 representative sites for detailed analysis
- Run full analysis on all 10 sites for completeness
- Consider restriction level filtering (>1) as sensitivity analysis
- Document all data preprocessing decisions
- Validate ensemble model on truly held-out future data

## Dependencies

See `requirements.txt` for full list:
- Core: pandas, numpy, scikit-learn
- ML: xgboost, catboost
- Stats: statsmodels
- Viz: matplotlib, seaborn
- AWS: boto3, sagemaker
