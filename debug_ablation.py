"""
Debug script to investigate negative R² issue
"""
import pandas as pd
import boto3

def debug_data(target_site='Lower Hutt'):
    """Check data quality and target variable"""
    s3 = boto3.client('s3')
    key = f"TrainingData/{target_site}.csv"
    obj = s3.get_object(Bucket='niwa-water-demand-modelling', Key=key)
    df = pd.read_csv(obj['Body'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"=== Data Debug for {target_site} ===\n")
    
    # Check columns
    print("Available columns:")
    print(df.columns.tolist())
    print()
    
    # Check target column
    target_col = target_site.replace(")", "").replace("(", "").replace(" ", "")
    print(f"Looking for target: '{target_col}'")
    print(f"Target in columns: {target_col in df.columns}")
    if target_site in df.columns:
        print(f"Original name '{target_site}' found in columns")
    print()
    
    # Check data split
    train = df[df['Date'] <= '2019-12-31']
    test = df[df['Date'] > '2019-12-31']
    print(f"Train size: {len(train)} ({train['Date'].min()} to {train['Date'].max()})")
    print(f"Test size: {len(test)} ({test['Date'].min()} to {test['Date'].max()})")
    print()
    
    # Check target statistics
    actual_target = None
    if target_col in df.columns:
        actual_target = target_col
    elif target_site in df.columns:
        actual_target = target_site
    
    if actual_target:
        print(f"Target '{actual_target}' statistics:")
        print(f"  Train mean: {train[actual_target].mean():.2f}, std: {train[actual_target].std():.2f}")
        print(f"  Test mean: {test[actual_target].mean():.2f}, std: {test[actual_target].std():.2f}")
        print(f"  Train NaN: {train[actual_target].isna().sum()}")
        print(f"  Test NaN: {test[actual_target].isna().sum()}")
    else:
        print(f"WARNING: Target column not found!")
    print()
    
    # Check restriction level distribution
    if 'Restriction level' in df.columns:
        print("Restriction level analysis:")
        print(f"  Train restriction distribution:")
        print(train['Restriction level'].value_counts().sort_index().to_string())
        print(f"  Train mean restriction: {train['Restriction level'].mean():.2f}")
        print(f"  Train days with restriction > 1: {(train['Restriction level'] > 1).sum()} ({(train['Restriction level'] > 1).sum()/len(train)*100:.1f}%)")
        print()
        print(f"  Test restriction distribution:")
        print(test['Restriction level'].value_counts().sort_index().to_string())
        print(f"  Test mean restriction: {test['Restriction level'].mean():.2f}")
        print(f"  Test days with restriction > 1: {(test['Restriction level'] > 1).sum()} ({(test['Restriction level'] > 1).sum()/len(test)*100:.1f}%)")
        print()
        
        # Check if restriction explains target difference
        if actual_target:
            train_no_restrict = train[train['Restriction level'] <= 1]
            test_no_restrict = test[test['Restriction level'] <= 1]
            print(f"  Target statistics (excluding restriction > 1):")
            print(f"    Train mean: {train_no_restrict[actual_target].mean():.2f}")
            print(f"    Test mean: {test_no_restrict[actual_target].mean():.2f}")
            print(f"    Difference: {abs(train_no_restrict[actual_target].mean() - test_no_restrict[actual_target].mean()):.2f}")
    else:
        print("WARNING: 'Restriction level' column not found!")
    print()
    
    # Check feature availability
    expected_features = ['doy', 'month', 'mday', 'wday', 'Season', 'is_holiday',
                        'Tmax', 'Sun', 'PE', 'Aquifer', 'API', 'storage',
                        'Restriction level',
                        f'{target_site} fTemp', f'{target_site} fPrecp',
                        f'{target_site} cm', f'{target_site} stat']
    print("Feature availability:")
    for feat in expected_features:
        available = feat in df.columns
        print(f"  {feat}: {'✓' if available else '✗'}")
    
    return df

if __name__ == "__main__":
    sites = [
        'Lower Hutt',
        'Wellington High Moa',
        'North Wellington Porirua',
        'Petone',
        'Porirua',
        'Upper Hutt',
        'Wainuiomata',
        'North Wellington Moa',
        'Wellington High Western',
        'Wellington Low Level'
    ]
    
    for site in sites:
        try:
            df = debug_data(site)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"ERROR processing {site}: {e}")
            print("\n" + "="*80 + "\n")
