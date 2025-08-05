import os
import s3fs
import pandas as pd
import time
import numpy as np
import pandas as pd
import math
import holidays
import warnings
import argparse
import boto3
from collections import defaultdict

warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"]=""
'''
How to run this script
- cd water-demand-prediction
- run: 
python scripts/prep_simulation_data.py --file_dir='s3://niwa-water-demand-modelling/SimulationInput/Final_HydroClimaticFile_ACCESS-CM2_ssp585FIX.csv'
- open autopilot_demand_simulation_inference_c5m5xlarge.ipynb and run all cells
- open consolidate_simulation_results.ipynb and run all cells

The final results will be saved to: s3://niwa-water-demand-modelling/Simulation/results/Final_HydroClimaticFile_EC-Earth3_ssp585FIX_full_results.csv
'''


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
    df_coeff = pd.read_csv(f"../data/{y_col} xpARA.csv")
    C1 = df_coeff["x1"].values[0]
    C2 = df_coeff["x2"].values[0]
    C3 = df_coeff["x3"].values[0]
    C4 = df_coeff["x4"].values[0]
    CP1 = df_coeff["x7"].values[0]
    CP2 = df_coeff["x8"].values[0]
    return C1, C2, C3, C4, CP1, CP2

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Simulation data generation script")
    parser.add_argument(
        "--file_dir",
        type=str,
        help="csv file path for simulation data."
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
    
    # read in restriction level data
    df_restrict = pd.read_csv("../data/RestrictionLevel.csv")
    df_restrict['Date'] = pd.to_datetime(df_restrict['Date'],format="%d/%m/%Y")
    df_restrict = df_restrict.set_index("Date")
    # read in cm data
    df_cm = pd.read_csv("../data/cm.csv")
    # define save directory
    save_dir = "data/simulation"
    simulation_name = file_dir.split("/")[-1].split(".")[0]
    save_dir_simulation = os.path.join(save_dir, simulation_name)
    os.makedirs(save_dir_simulation, exist_ok=True)
    
    # input data
#     df = pd.read_csv(file_dir)
#     file_path = "/nesi/project/niwa03661/residential_water_pcd/data_20241219/Final_HydroClimaticFile_ACCESS-CM2_ssp126.csv"
    header_nrows = 49 # nrows after which the real tabular data begins
    header_skiprows = 5 # first few rows which contains metadata only such as file location
    # therefore, the contents in header_nrows-header_skiprows are the real headers to be used for tabular data

    # read headers into dataframe
    header = pd.read_csv(file_dir, skiprows=header_skiprows, nrows=header_nrows-header_skiprows)
    # for each row, extract the header names, and save it to the list
    header = [e[0].strip() for e in header.values]
    # add in 3 more items in front of the extracted headers ["replicate", "Day of Year", "year"]
    # now the header length is the same as the comma separated tabular data content
    header = ["replicate", "Day of Year", "year"] + header
    # read the tabular data into dataframe, use the generated header as the column names
    data = pd.read_csv(file_dir, skiprows=header_nrows+1, names=header)
    
    # rename some of the features to what we want
    rename_dict = {
    "Aquifer Recharge 12.":"Aquifer",
    "Potential Evaporatio":"PE",
    "Max temp":"Tmax",
    "SunshineHrs":"Sun",
    # "WVRain":"WVRain",
    "KelburnRain":"KerburnRain"
    }

    # quick way to generate Date column and make it into datetime format
    data["Date"] = pd.to_datetime(data["year"] * 1000 + data["Day of Year"], format="%Y%j")
    data = data.rename(columns=rename_dict)
    pcd_cols = [e for e in data.columns if "PCD_" in e]
    new_cols = []
    
    # rename the statistical results columns
    for pcd_col in pcd_cols:
        new_col = pcd_col.replace("PCD_", "") # remove PCD from the column names
        new_col = new_col.replace("_", " ") # remove underscore from the column names
#         new_col = new_col + " stat"
        new_cols.append(new_col)
        data.rename(columns={pcd_col:new_col}, inplace=True)
    
    # group by replicate number for the data prep
    replicate_groups = data.groupby(["replicate"])
    d = defaultdict(list)
    for group in replicate_groups.groups:
        df = replicate_groups.get_group(group)
        df, df1 = preprocess(df)
        df1["replicate"] = group

        for y_col in y_cols:
            df_i = df1.copy() # general features for all sites
            df_s = df.copy() # site specific statistical input feature
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
            # remove ( and ) from original y_col to match with simulation input data
            y_col = y_col.replace(")", "")
            y_col = y_col.replace("(", "")
            df_s = df_s.rename(columns={y_col:y_col+" stat"})
            df_i = df_i.join(df_s[y_col+" stat"], how="left")
            df_hist = df_i.join(df_restrict)
            df_hist["Restriction level"] = df_hist["Restriction level"].fillna(value=0)
            df_hist.columns = [e.replace(")", "") for e in df_hist.columns]
            df_hist.columns = [e.replace("(", "") for e in df_hist.columns]
            # write replicate to corresponding y_col list
            df_hist = df_hist.reset_index()
            # move replicate to the front
            col = df_hist.pop("replicate")
            df_hist.insert(0, col.name, col)
            d[y_col].append(df_hist)
    
    # wrap up all the target dataframe list
    for y_col in new_cols:
        filename = y_col.replace(")", "")
        filename = filename.replace("(", "")
        foldername = filename.replace(" ", "")
        s3_folder = "Simulation"
        df_out = pd.concat(d[y_col], axis=0) # get all replicates appended
        s3_file_path = f"s3://{bucket_name}/{s3_folder}/{foldername}/{filename}.csv"
        df_out.to_csv(s3_file_path, index=False)
        # file_path_simulation= os.path.join(
        #     save_dir_simulation,
        #     f"{filename}.csv"
        # )

        # df_out.to_csv(file_path_simulation, index=False)
        # print(f"saved to local: {file_path_simulation}")
        # s3.meta.client.upload_file(Filename=file_path_simulation, Bucket=bucket_name, Key=f"{s3_folder}/{foldername}/{filename}.csv")
        print(f"uploaded to s3 bucket {bucket_name}: {s3_folder}/{foldername}/{filename}.csv")