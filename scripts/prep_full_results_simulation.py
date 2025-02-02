import glob
import time
import os
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stitch Simulation Results")
    parser.add_argument(
        "--result_dir",
        type=str,
        help="results directory containing all the simulation files",
        default="/nesi/project/niwa03661/residential_water_pcd/inference/simulation/20250107/Final_HydroClimaticFile_ACCESS-CM2_ssp126/results"
    )
    args, unknown = parser.parse_known_args()
    result_dir = args.result_dir
#     result_dir = "/nesi/project/niwa03661/residential_water_pcd/inference/results"
    y_cols = ['Lower Hutt', 'Petone',
           'Wainuiomata', 'Upper Hutt', 'Porirua', 'Wellington High (Moa)',
           'Wellington High (Western)', 'Wellington Low Level',
           'North Wellington (Moa)', 'North Wellington (Porirua)']
    y_cols = [e.replace(")", "") for e in y_cols]
    y_cols = [e.replace("(", "") for e in y_cols]
#     df_y = pd.read_csv("/nesi/project/niwa03661/residential_water_pcd/Residential_water_pcd_2006_2024_Y.csv")
#     df_y.columns = [e.replace(")", "") for e in df_y.columns]
#     df_y.columns = [e.replace("(", "") for e in df_y.columns]
#     df_y.index = pd.to_datetime(df_y["Date"])
#     curr_time = time.strftime("%Y%m%d", time.localtime())
#     output_dir = os.path.join(result_dir, curr_time)
#     os.makedirs(output_dir, exist_ok=True)
    # find all result files
    result_files = glob.glob(f"{result_dir}/*full_prediction.csv")
    df_list = []

#     for y_col in y_cols:
    for file in result_files:
        df = pd.read_csv(file)
        col = df.columns[0] # get pred column name
        # check which y_col
#         df  = df[["Date", col]].set_index("Date")
        if "replicate" in df.columns:
            rep_unique = df["replicate"].unique()
            # check if only 1 replicate
            if len(rep_unique)>1:
                # include replicate as index
                df  = df[["Date", "replicate", col]].set_index(["replicate", "Date"])
            else:
                df  = df[["Date", col]].set_index("Date")
        else:
            df  = df[["Date", col]].set_index("Date")
            
        df_list.append(df)
        print(f"retrieved {col} results")

    df_ps = pd.concat(df_list, axis=1)
    df_ps[y_cols].to_csv(os.path.join(result_dir, "full_results.csv"))