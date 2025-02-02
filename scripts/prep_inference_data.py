import sys
import os
import re
import pandas as pd
import glob
import time
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
# from tensorflow import keras
import matplotlib.pyplot as plt
import csv
import math
import sklearn.metrics as skm
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, GRU, SimpleRNN, Bidirectional, Dropout
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as mticker
import joblib
from prophet import Prophet
from pandas.plotting import autocorrelation_plot
import holidays
import warnings
import argparse
import boto3

warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"]=""



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

def calc_fTemp(Tmax, C1, C2, C3, C4):
    return (C1 + C2 * np.tanh((Tmax - C3) / C4))

def calc_fPrecp(KerburnRain_rollmean3, CP1, CP2):
    return(1 - CP1 * (1 - np.exp(-CP2 * KerburnRain_rollmean3)))

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

def read_coefficient(y_col):
    df_coeff = pd.read_csv(f"/nesi/project/niwa03661/residential_water_pcd/data/{y_col} xpARA.csv")
    C1 = df_coeff["x1"].values[0]
    C2 = df_coeff["x2"].values[0]
    C3 = df_coeff["x3"].values[0]
    C4 = df_coeff["x4"].values[0]
    CP1 = df_coeff["x7"].values[0]
    CP2 = df_coeff["x8"].values[0]
    return C1, C2, C3, C4, CP1, CP2

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation script")
    parser.add_argument(
        "--file_dir",
        type=str,
        help="csv file directory."
    )
    args, unknown = parser.parse_known_args()
    file_dir = args.file_dir
    curr_time = time.strftime("%Y%m%d", time.localtime())
    s3 = boto3.resource('s3')
    bucket_name = 'niwa-water-demand-modelling'
    print(curr_time)
    y_cols = ['Lower Hutt', 'Petone',
       'Wainuiomata', 'Upper Hutt', 'Porirua', 'Wellington High (Moa)',
       'Wellington High (Western)', 'Wellington Low Level',
       'North Wellington (Moa)', 'North Wellington (Porirua)']
    
    # input data
    df = pd.read_csv(file_dir)

    # # target data
    df_Y = pd.read_csv("/nesi/project/niwa03661/residential_water_pcd/Residential_water_pcd_2006_2024_Y.csv")
    df_Y['Date'] = pd.to_datetime(df_Y['Date'])
    df_Y = df_Y.set_index("Date")

    df, df1 = preprocess(df)
    df_restrict = pd.read_csv("/nesi/project/niwa03661/residential_water_pcd/data/RestrictionLevel.csv")
    df_restrict['Date'] = pd.to_datetime(df_restrict['Date'],format="%d/%m/%Y")
    df_restrict = df_restrict.set_index("Date")
    # df_train = exclude_restrictions(df.set_index("Date"), df_restrict.set_index("Date"))
    df_cm = pd.read_csv("/nesi/project/niwa03661/residential_water_pcd/data/cm.csv")
    save_dir_historical = "/nesi/project/niwa03661/residential_water_pcd/inference/historical"
    save_dir_post_training = "/nesi/project/niwa03661/residential_water_pcd/inference/post_training"
    train_start_dt = "2006-01-01"
    train_end_dt = "2024-08-31"
#     os.makedirs(os.path.join(save_dir, curr_time), exist_ok=True)
    y_cols_new = []
    # df_hist = pd.DataFrame()
    for y_col in y_cols:
        df_i = df1.copy() # general features for all sites
        df_s = df.copy() # site specific statistical input feature
        df_y = df_Y.copy()
        C1, C2, C3, C4, CP1, CP2 = read_coefficient(y_col)
        fTemp = calc_fTemp(df1["Tmax"], C1, C2, C3, C4)
        KerburnRain_rollmean3 = df1.loc[:,'KerburnRain'].rolling(window=3).mean().fillna(0)
        fPrecp = calc_fPrecp(KerburnRain_rollmean3, CP1, CP2)
        df_i[f"{y_col} fTemp"] = fTemp
        df_i[f"{y_col} fPrecp"] = fPrecp
        df_cm1 = df_cm.reset_index()[["month", y_col]].rename(columns={y_col: f"{y_col} cm"})
        df_i = pd.merge(df_i, df_cm1, on=["month"], how="left")
        df_i = df_i.set_index("Date")
        df_s = df_s.set_index("Date")
    #     df_y = df_y.set_index("Date")
        df_s = df_s.rename(columns={y_col:y_col+" stat"})
        df_i = df_i.join(df_s[y_col+" stat"], how="left")

        df_hist = df_i.join(df_y[y_col], how="left")
        df_hist = df_hist.join(df_restrict)
        df_hist["Restriction level"] = df_hist["Restriction level"].fillna(value=0)
        df_hist[y_col] = df_hist[y_col].fillna(value=0)
        df_hist.columns = [e.replace(")", "") for e in df_hist.columns]
        df_hist.columns = [e.replace("(", "") for e in df_hist.columns]
        filename = y_col.replace(")", "")
        filename = filename.replace("(", "")
        foldername = filename.replace(" ", "")
        file_path_post_training = os.path.join(
            save_dir_post_training,
            f"{filename}.csv"
        )
        file_path_historical = os.path.join(
            save_dir_historical,
            f"{filename}.csv"
        )
        df_hist = df_hist.reset_index()
        df_hist["Date"] = pd.to_datetime(df_hist["Date"])
        df_historical = df_hist[df_hist["Date"] < pd.to_datetime(train_start_dt)]
        df_post_training = df_hist[df_hist["Date"] > pd.to_datetime(train_end_dt)]
        
        df_historical["Date"] = df_historical["Date"].apply(lambda x: datetime.strftime(x, "%d/%m/%Y"))
        if not os.path.exists(file_path_historical):
            df_historical.set_index("Date").to_csv(file_path_historical)
        
        df_post_training["Date"] = df_post_training["Date"].apply(lambda x: datetime.strftime(x, "%d/%m/%Y"))
        df_post_training.set_index("Date").to_csv(file_path_post_training)
        # find last timestamp in the folder, if no file, look for filename in the main s3 bucket
        # only keep data > last timestamp
        # if no data, don't update the folder

        s3.meta.client.upload_file(Filename=file_path_post_training, Bucket=bucket_name, Key=f"{foldername}/{filename}.csv")