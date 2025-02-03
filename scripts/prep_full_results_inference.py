import glob
import time
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stitch Inference Results")
    parser.add_argument(
        "--result_dir",
        type=str,
        help="results directory containing all the inference files",
        default="/nesi/project/niwa03661/residential_water_pcd/inference/results"
    )
    args, unknown = parser.parse_known_args()
    result_dir = args.result_dir
#     result_dir = "/nesi/project/niwa03661/residential_water_pcd/inference/results"
    y_cols = ['Lower Hutt', 'Petone',
           'Wainuiomata', 'Upper Hutt', 'Porirua', 'Wellington High (Moa)',
           'Wellington High (Western)', 'Wellington Low Level',
           'North Wellington (Moa)', 'North Wellington (Porirua)']
    df_y = pd.read_csv("/nesi/project/niwa03661/residential_water_pcd/Residential_water_pcd_2006_2024_Y.csv")
    df_y.columns = [e.replace(")", "") for e in df_y.columns]
    df_y.columns = [e.replace("(", "") for e in df_y.columns]
    df_y.index = pd.to_datetime(df_y["Date"])
    curr_time = time.strftime("%Y%m%d", time.localtime())
    output_dir = os.path.join(result_dir, curr_time)
    os.makedirs(output_dir, exist_ok=True)
    df_list = []

    for y_col in y_cols:
        y_col_1 = y_col.replace(")", "")
        col = y_col_1.replace("(", "")
        y_col_1 = col.replace(" ", "")
        hist_file = glob.glob(f"{result_dir}/{y_col_1}_historical_pred.csv")[0]
        hist_pred = pd.read_csv(hist_file)
        train_file = glob.glob(f"{result_dir}/{y_col_1}_training_pred.csv")[0]
        train_pred = pd.read_csv(train_file)
        posttrain_file = glob.glob(f"{result_dir}/full_results.csv")[0]
        posttrain_pred = pd.read_csv(posttrain_file)
        posttrain_pred = posttrain_pred[["Date", col]]
        df_pred = pd.concat([hist_pred, train_pred, posttrain_pred], axis=0)
        df_pred.index = pd.to_datetime(df_pred["Date"], format="mixed", dayfirst=True)
        df_list.append(df_pred[df_pred.columns[0]])

        # check results
        df_hist_1 = df_pred[[col, "Restriction level"]].join(df_y[[col]].rename(columns={col: col+"_obs"}), how="left")
        df_hist_2 = df_hist_1[(df_hist_1.index>"2022-12-31")]
#         plt.scatter(df_hist_2[col], df_hist_2[col+"_obs"])
#         plt.title(col)
#         plt.show()
        df_hist_1 = df_hist_1[(df_hist_1.index>"2022-12-31")&(df_hist_1.index<"2025-01-01")]
        # df_hist_1.index = pd.to_datetime(df_hist_1.index)
        fig, ax = plt.subplots()
        df_hist_1[[col+"_obs", col]].plot(figsize=(13,4), title=col, alpha=0.5, ax=ax)
        ax1 = ax.twinx()
        df_hist_1[["Restriction level"]].plot(figsize=(13,4), title=col, color='r', alpha=0.5, ax=ax1)
        ax1.legend(loc='upper left')
        plt.savefig(os.path.join(output_dir, f"{y_col_1}.png"))
        print(f"save {y_col} results to {output_dir}")

    df_ps = pd.concat(df_list, axis=1)
    df_ps.to_csv(os.path.join(output_dir, "full_results.csv"))