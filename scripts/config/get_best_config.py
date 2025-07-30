import os
import pandas as pd
import sys
import figurePlotter
from scripts.plot.process_result import process_two_stage
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", 40)
# pd.set_option("display.width", 1000)

if len(sys.argv) > 2:
    results_csv_path = sys.argv[1]
    results_csv_path_tuned = sys.argv[2]
else:
    results_csv_path = os.path.join(os.environ["BITGEN_ROOT"], "results/csv/exec_tune_arpg_1input.csv") 
    results_csv_path_tuned = os.path.join(os.environ["BITGEN_ROOT"], "results/csv/exec_tune_best_arpg_1input.csv")

print("Read results from: ", results_csv_path)


df = pd.read_csv(results_csv_path)
df = process_two_stage(df=df)
print(df)

df_best = pd.DataFrame()
apps = df["app"].unique()
for app in apps:
    app_df = df[df["app"] == app]
    print(app_df)
    max_throughput_row = app_df.loc[app_df["throughput"].idxmax()]
    print(max_throughput_row)
    df_best = pd.concat([df_best, max_throughput_row.to_frame().T], ignore_index=True)

print("Best configurations:")
print(df_best)
df_best.to_csv(results_csv_path_tuned, index=False)
