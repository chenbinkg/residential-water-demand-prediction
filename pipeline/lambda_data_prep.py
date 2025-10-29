import json
import boto3
import pandas as pd
import numpy as np
import math
from datetime import datetime
import holidays
from io import StringIO

def MonthToSeason(x):   
    global season
    if x == 3 or x == 4 or x == 5:
         season = 1
    elif x == 6 or x == 7 or x == 8:
         season = 2
    elif x == 9 or x == 10 or x == 11:
         season = 3
    elif x == 12 or x == 1 or x == 2:
         season = 4
    else:
         season = np.nan 
    return season

def is_holiday(x, holiday_region):
    if x in holiday_region:
        return 1
    else:
        return 0

def calc_fTemp(Tmax, C1, C2, C3, C4):
    return (C1 + C2 * np.tanh((Tmax - C3) / C4))

def calc_fPrecp(KerburnRain_rollmean3, CP1, CP2):
    return(1 - CP1 * (1 - np.exp(-CP2 * KerburnRain_rollmean3)))

def read_coefficient_from_s3(s3, bucket_name, y_col):
    key = f"data/{y_col} xpARA.csv"
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    data = obj['Body'].read().decode('utf-8')
    df_coeff = pd.read_csv(StringIO(data))
    return df_coeff["x1"].values[0], df_coeff["x2"].values[0], df_coeff["x3"].values[0], df_coeff["x4"].values[0], df_coeff["x7"].values[0], df_coeff["x8"].values[0]

def preprocess(df, corr_matrix=False):
    
    # dCentre=cols[tp] 
    dCentre = 'Lower Hutt'
    print('data processing')
    # df = df[['Date','Day of Year', dCentre, 'Aquifer','PE', 'Tmax', 'Sun', 'WVRain', 'KerburnRain','PCD_Wellington_High_Moa', 'PCD_Wellington_Low_Level','PCD_North_Wellington_Moa']]
    df['Aquifer'] = df['Aquifer']/5     # division with 5 is due to conversion in hydrolcimatic file. 
    df['WVRain'] = df['WVRain']/21
    if df['Date'].dtype == "O":
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df['Days'] = pd.to_datetime(df['Date'],format="%d/%m/%Y")

    # i+7 = sum(I,i+1,i+2,…,i+6)
    df['Rain_L7DAYS'] = df.loc[:,'Aquifer'].rolling(window=7).sum() - df['Aquifer']    # window =7  to window 5
    df['Rain_L6DAYS'] = df.loc[:,'Aquifer'].rolling(window=6).sum() - df['Aquifer']    # window =7  to window 5
    df['Rain_L5DAYS'] = df.loc[:,'Aquifer'].rolling(window=5).sum() - df['Aquifer']    # window =7  to window 5
    df['Rain_L4DAYS'] = df.loc[:,'Aquifer'].rolling(window=4).sum() - df['Aquifer']    # window =7  to window 5
    df['Rain_L3DAYS'] = df.loc[:,'Aquifer'].rolling(window=3).sum() - df['Aquifer']    # window =7  to window 5
    df['Rain_L2DAYS'] = df.loc[:,'Aquifer'].rolling(window=2).sum() - df['Aquifer']    # window =7  to window 5

    df = df.fillna(df.mean())

    df['Season'] = df['Days'].dt.month.apply(lambda x : MonthToSeason(x))
    
    for i in df.columns:
        if df[i].isnull().sum().sum() >= df.shape[0]/2:
            df = df.drop(columns=i) 
    
    df = df.fillna(df.mean())
    df = df.reset_index(drop=True)
    df = df.dropna()
    df = df.reset_index(drop=True)

    df['Rainlag1'] = df['Aquifer'].shift(1)
    df['Rainlag1'] = df['Rainlag1'].fillna(method="bfill")
#     df['Rainlag1'] = df['Rainlag1'].fillna(df['Rainlag1'].mean()) 

    df['Rainlag2'] = df['Aquifer'].shift(2)
    df['Rainlag2'] = df['Rainlag2'].fillna(method="bfill")
#     df['Rainlag2'] = df['Rainlag2'].fillna(df['Rainlag2'].mean())
    
    df['Rainlag3'] = df['Aquifer'].shift(3)
    df['Rainlag3'] = df['Rainlag3'].fillna(method="bfill")
#     df['Rainlag3'] = df['Rainlag3'].fillna(df['Rainlag3'].mean()) 


    df['PElag1'] = df['PE'].shift(1)
    df['PElag1'] = df['PElag1'].fillna(method="bfill")
#     df['PElag1'] = df['PElag1'].fillna(df['PElag1'].mean()) 
    df['PElag2'] = df['PE'].shift(2)
    df['PElag2'] = df['PElag2'].fillna(method="bfill")
#     df['PElag2'] = df['PElag2'].fillna(df['PElag2'].mean()) 
    df['PElag3'] = df['PE'].shift(3)
    df['PElag3'] = df['PElag3'].fillna(method="bfill")
#     df['PElag3'] = df['PElag3'].fillna(df['PElag3'].mean()) 

    df['Tmaxlag1'] = df['Tmax'].shift(1)
    df['Tmaxlag1'] = df['Tmaxlag1'].fillna(method="bfill")
    df['Tmaxlag2'] = df['Tmax'].shift(2)
    df['Tmaxlag2'] = df['Tmaxlag2'].fillna(method="bfill")
    df['Tmaxlag3'] = df['Tmax'].shift(3)
    df['Tmaxlag3'] = df['Tmaxlag3'].fillna(method="bfill")
    
    df['Sunlag1'] = df['Sun'].shift(1)
    df['Sunlag1'] = df['Sunlag1'].fillna(method="bfill")
    df['Sunlag2'] = df['Sun'].shift(2)
    df['Sunlag2'] = df['Sunlag2'].fillna(method="bfill")
    df['Sunlag3'] = df['Sun'].shift(3)
    df['Sunlag3'] = df['Sunlag3'].fillna(method="bfill")

    cs = np.cos(2*math.pi*((df['Day of Year'] % 365)/365))
    cw = np.cos(2*math.pi*((df['Day of Year'] % 7)/7))

    df['ANcyc'] =  np.sign(cs)*np.abs(cs)   
    #np.cos(2*np.pi*((df['Days'] % 365)/365)) 
    #df['Weekcyc'] = np.sign(cw)*np.abs(cw)  

    alpha=0.8
    df['API'] = np.zeros(len(df))
    #print(df)
    for i in range(1,len(df)):
        df.at[i,'API']=alpha*df.at[i-1,'API']+df.at[i,'Aquifer']
    #v=0.95*0.0+11.2
    #v1=0.95*v+5

    df['lagAPI'] = df['API'].shift(1)
    df['lagAPI'] = df['lagAPI'].fillna(method="bfill")
#     df['lagAPI'] = df['lagAPI'].fillna(df['lagAPI'].mean())

    K=0.2
    df['storage'] = np.zeros(len(df))
    #print(df)
    C1=math.exp(-K)
    for i in range(1,len(df)):
        df.at[i,'storage']=C1*df.at[i-1,'storage']+((1-C1)*df.at[i,'Aquifer']/K)
    #v=0.95*0.0+11.2
    #v1=0.95*v+5
    df['Storagelag1'] = df['storage'].shift(1)
    df['Storagelag1'] = df['Storagelag1'].fillna(method="bfill")
#     df['Storagelag1'] = df['Storagelag1'].fillna(df['Storagelag1'].mean())

    nz_wgn_holidays = holidays.country_holidays('NZ', subdiv='WGN')
    df['wday'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['mday'] = df['Date'].dt.days_in_month
    df["doy"] = df['Date'].dt.dayofyear
    df["is_holiday"] = df["Date"].apply(lambda x: is_holiday(x, nz_wgn_holidays))
    df['KerburnRain_rollmean'] = df.loc[:,'KerburnRain'].rolling(window=3).mean()

    # dfdate = df
    features = ['Date', 'doy', 'Aquifer', 'PE', 'month', 'mday',
           'is_holiday', 'wday', 'API', 'lagAPI', 
                'Tmax', "Tmaxlag1", 
           'Sun', "Sunlag1", 
                'WVRain', 'KerburnRain', 'Rain_L7DAYS',
           'Rain_L6DAYS', 'Rain_L5DAYS', 'Rain_L4DAYS', 'Rain_L3DAYS',
           'Rain_L2DAYS', 'Season', 'Rainlag1', 'Rainlag2', 'Rainlag3', 'PElag1', 'PElag2',
           'PElag3', 'ANcyc', 'storage', 'Storagelag1']
    df1 = df[features]
    # df1 = df.drop(columns=[dCentre,'Date','Days','lag','Season'])
    df1 = df1.fillna(df1.mean())
    
    if corr_matrix:
        matrix = df.corr()
        corelation=matrix[dCentre]
        percentageGap=100*pd.isna(df).sum()/len(df)

        gh=pd.DataFrame()
        pd.options.display.float_format = '{:,.2f}'.format
        gh['P']= percentageGap[1:]
        gh['C']= corelation
        gh.to_csv("Gap_Correlation.csv",float_format='%.3f')

    return df, df1


def exclude_restrictions(df, df_restrict):
    df_train = df.join(df_restrict, how="left")
    df_train = df_train[df_train["Restriction level"]<=1]
    del df_train["Restriction level"]
    return df_train

def calc_cm(df, y_cols):
    # apply after restriction exclusion
    df["year-month"] = df["year"].astype(str) + "-" + df["month"].astype(str)
    m_counts = df.groupby(["year-month"])["month"].count()
    m_counts_full = m_counts[m_counts>=28].index
    df = df[df["year-month"].isin(m_counts_full)]
    df_cm = df.groupby(["month"])[y_cols].mean()
    for y_col in y_cols:
        df_cm[y_col] = df_cm[y_col]/df[y_col].mean()
    return df_cm


def lambda_handler(event, context):
    bucket_name = event['bucket_name']
    file_key = event['file_key']
    s3 = boto3.client('s3')
    
    y_cols = ['Lower Hutt', 'Petone', 'Wainuiomata', 'Upper Hutt', 'Porirua', 
              'Wellington High (Moa)', 'Wellington High (Western)', 'Wellington Low Level',
              'North Wellington (Moa)', 'North Wellington (Porirua)']
    
    try:
        # Read input file
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        
        # Read reference data
        df_Y = pd.read_csv(StringIO(s3.get_object(Bucket=bucket_name, Key="data/Residential_water_pcd_2006_2024_Y.csv")['Body'].read().decode('utf-8')))
        df_Y['Date'] = pd.to_datetime(df_Y['Date']).set_index('Date')
        
        df_restrict = pd.read_csv(StringIO(s3.get_object(Bucket=bucket_name, Key="data/RestrictionLevel.csv")['Body'].read().decode('utf-8')))
        df_restrict['Date'] = pd.to_datetime(df_restrict['Date'], format="%d/%m/%Y").set_index('Date')
        
        df_cm = pd.read_csv(StringIO(s3.get_object(Bucket=bucket_name, Key="data/cm.csv")['Body'].read().decode('utf-8')))
        
        df, df1 = preprocess(df)
        train_end_dt = "2024-08-31"
        processed_files = []
        
        for y_col in y_cols:
            df_i = df1.copy()
            C1, C2, C3, C4, CP1, CP2 = read_coefficient_from_s3(s3, bucket_name, y_col)
            
            fTemp = calc_fTemp(df1["Tmax"], C1, C2, C3, C4)
            KerburnRain_rollmean3 = df1.loc[:,'KerburnRain'].rolling(window=3).mean().fillna(0)
            fPrecp = calc_fPrecp(KerburnRain_rollmean3, CP1, CP2)
            
            df_i[f"{y_col} fTemp"] = fTemp
            df_i[f"{y_col} fPrecp"] = fPrecp
            
            df_cm1 = df_cm.reset_index()[["month", y_col]].rename(columns={y_col: f"{y_col} cm"})
            df_i = pd.merge(df_i, df_cm1, on=["month"], how="left").set_index("Date")
            
            df_hist = df_i.join(df_Y[y_col], how="left").join(df_restrict)
            df_hist["Restriction level"] = df_hist["Restriction level"].fillna(0)
            df_hist[y_col] = df_hist[y_col].fillna(0)
            
            df_hist.columns = [e.replace(")", "").replace("(", "") for e in df_hist.columns]
            filename = y_col.replace(")", "").replace("(", "")
            foldername = filename.replace(" ", "")
            
            df_post_training = df_hist.reset_index()
            df_post_training["Date"] = pd.to_datetime(df_post_training["Date"])
            df_post_training = df_post_training[df_post_training["Date"] > pd.to_datetime(train_end_dt)]
            df_post_training["Date"] = df_post_training["Date"].apply(lambda x: datetime.strftime(x, "%d/%m/%Y"))
            
            s3_file_path = f"InferenceData/{foldername}/{filename}.csv"
            csv_buffer = StringIO()
            df_post_training.set_index("Date").to_csv(csv_buffer)
            s3.put_object(Bucket=bucket_name, Key=s3_file_path, Body=csv_buffer.getvalue())
            processed_files.append(s3_file_path)
        
        return {
            'statusCode': 200,
            'processed_files': processed_files,
            'bucket_name': bucket_name
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'error': str(e)
        }