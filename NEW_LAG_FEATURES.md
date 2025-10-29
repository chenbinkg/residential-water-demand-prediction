# New Lag Features Added Based on PACF Analysis

## Summary of Changes

### 1. New Lag Features Added

Based on PACF analysis results, the following lag features have been added:

| Variable | New Features | PACF Justification |
|----------|--------------|-------------------|
| **Target** | Targetlag1, Targetlag2, Targetlag3 | Captures autoregressive demand patterns |
| **API** | lag2API, lag4API | PACF: -0.351 (lag 2), -0.156 (lag 4) |
| **Tmax** | Tmaxlag2, Tmaxlag3 | PACF: 0.112 (lag 2), 0.078 (lag 3) |
| **Sun** | Sunlag2, Sunlag3 | PACF: -0.273 (lag 2), -0.211 (lag 3) |
| **storage** | Storagelag2, Storagelag4 | PACF: -0.346 (lag 2), -0.152 (lag 4) |

### 2. Updated Feature Groups for Ablation Study

New groups added to test incremental impact:

```python
'tmax_lag2_lag3': ['Tmaxlag2', 'Tmaxlag3']      # After 'tmax_lag'
'sun_lag2_lag3': ['Sunlag2', 'Sunlag3']          # After 'sun_lag'
'api_lag2_lag4': ['lag2API', 'lag4API']          # After 'api'
'storage_lag2_lag4': ['Storagelag2', 'Storagelag4']  # After 'storage'
'target_lag': ['Targetlag1', 'Targetlag2', 'Targetlag3']  # At the end
```

### 3. Data Processing Flow

1. **Load from S3**: Original training data
2. **Add lag features**: Automatically adds all new lags
3. **Save locally**: Saved to `data/processed/{site}_with_lags.csv`
4. **Run experiments**: Uses enhanced dataset

### 4. Files Modified

- `scripts/add_lag_features.py` (NEW): Helper function to add lags
- `scripts/feature_analysis.py`: Updated feature groups
- `run_experiments.py`: Updated prepare_data() function

## Usage

### Run Experiments with New Features

```bash
# Run all experiments with new lag features
python run_experiments.py

# Run ACF analysis only
python run_experiments.py --acf

# Run ablation study only
python run_experiments.py --ablation
```

### Check Processed Data

After running, check the processed data files:
```bash
ls -lh data/processed/
# You'll see files like:
# Lower_Hutt_with_lags.csv
# Wellington_High_Moa_with_lags.csv
# etc.
```

## Expected Ablation Study Output

The ablation study will now show incremental improvements:

```
feature_group       n_features  rmse    mae     r2
...
tmax_lag            X          XX.XX   XX.XX   X.XX
tmax_lag2_lag3      X+2        XX.XX   XX.XX   X.XX  ← Impact of Tmaxlag2-3
sun_lag             X          XX.XX   XX.XX   X.XX
sun_lag2_lag3       X+2        XX.XX   XX.XX   X.XX  ← Impact of Sunlag2-3
...
api                 X          XX.XX   XX.XX   X.XX
api_lag2_lag4       X+2        XX.XX   XX.XX   X.XX  ← Impact of lag2API, lag4API
storage             X          XX.XX   XX.XX   X.XX
storage_lag2_lag4   X+2        XX.XX   XX.XX   X.XX  ← Impact of Storagelag2-4
restriction         X          XX.XX   XX.XX   X.XX
target_lag          X+3        XX.XX   XX.XX   X.XX  ← Impact of target lags (LAST)
```

**Note**: Target lags are tested LAST and EXCLUDED from algorithm benchmark and feature importance studies.

## Justification for Paper

### Target Lags (1-3)
> "To capture autoregressive patterns in water demand, we included lagged demand features (Targetlag1-3). These features allow the model to learn day-to-day consumption persistence and weekly patterns."

### Extended Weather Lags
> "PACF analysis revealed significant direct effects beyond lag 1 for weather variables. We added Tmaxlag2-3 (PACF: 0.112, 0.078) and Sunlag2-3 (PACF: -0.273, -0.211) to capture multi-day weather influence on water demand behavior."

### Extended API/Storage Lags
> "The oscillating PACF patterns for API (lag 2: -0.351, lag 4: -0.156) and storage (lag 2: -0.346, lag 4: -0.152) suggested compensatory dynamics in moisture indices. We included these lags to capture the full cycle of wetness accumulation and depletion."

## Important Notes

### Target Lags - Special Handling

**Why target lags are excluded from final model:**
1. ❌ **Not available for future inference** - Can't predict tomorrow without knowing today's demand
2. ❌ **Not available for historical scenarios** - Pre-2006 data has no observations
3. ✅ **Only for ablation study** - Shows theoretical upper bound of model performance

**What's included where:**
- ✅ **Ablation study**: Includes target_lag (last group) - shows impact
- ❌ **Algorithm benchmark**: Excludes target_lag - realistic comparison
- ❌ **Feature importance**: Excludes target_lag - operational features only

### Other Notes

- All lag features use forward-fill then backward-fill for missing values
- Local copies saved to `data/processed/` for verification
- Feature groups are cumulative in ablation study
- Target lags placed at end to show "ideal scenario" performance
