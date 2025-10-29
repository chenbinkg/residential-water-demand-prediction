"""
Add additional lag features based on PACF analysis
"""
import pandas as pd
import numpy as np

def add_lag_features(df, target_col, site_name=None):
    """Add new lag features based on PACF recommendations
    
    Args:
        df: DataFrame with original features
        target_col: Name of target variable
        site_name: Site name for site-specific features
        
    Returns:
        DataFrame with additional lag features
    """
    df = df.copy()
    
    # Target lags (1-3)
    for lag in [1, 2, 3]:
        df[f'Targetlag{lag}'] = df[target_col].shift(lag).ffill().bfill()
    
    # API lags (2, 4)
    if 'API' in df.columns:
        df['lag2API'] = df['API'].shift(2).ffill().bfill()
        df['lag4API'] = df['API'].shift(4).ffill().bfill()
    
    # Tmax lags (2, 3)
    if 'Tmax' in df.columns:
        df['Tmaxlag2'] = df['Tmax'].shift(2).ffill().bfill()
        df['Tmaxlag3'] = df['Tmax'].shift(3).ffill().bfill()
    
    # Sun lags (2, 3)
    if 'Sun' in df.columns:
        df['Sunlag2'] = df['Sun'].shift(2).ffill().bfill()
        df['Sunlag3'] = df['Sun'].shift(3).ffill().bfill()
    
    # Storage lags (2, 4)
    if 'storage' in df.columns:
        df['Storagelag2'] = df['storage'].shift(2).ffill().bfill()
        df['Storagelag4'] = df['storage'].shift(4).ffill().bfill()
    
    # Stat lags (1, 2, 3) - site-specific
    if site_name:
        stat_col = f'{site_name} stat'
        if stat_col in df.columns:
            for lag in [1, 2, 3]:
                df[f'{site_name} statlag{lag}'] = df[stat_col].shift(lag).ffill().bfill()
    else:
        if 'stat' in df.columns:
            for lag in [1, 2, 3]:
                df[f'statlag{lag}'] = df['stat'].shift(lag).ffill().bfill()
    
    return df
