import os
import pandas as pd
import figurePlotter

from process_result import process_two_stage



results_csv_path = os.path.join(
    os.environ["BITGEN_ROOT"], "results/csv/exec_opt_breakdown_arpg_1input.csv"
)

baseline_csv_path =os.path.join(
    os.environ["BITGEN_ROOT"], "results/csv/exec_baseline_arpg_1input.csv"
)


def load_csv(results_path):
    df = pd.read_csv(results_path)
    df = process_two_stage(df=df)
    df = df[df["name"] == "run_regex"]
    df = df[["exe", "app", "throughput"]]
    return df

df = load_csv(results_csv_path)
df_baseline = pd.read_csv(baseline_csv_path)
df = pd.concat([df, df_baseline], ignore_index=True) 
df_pivot = df.pivot(index="app", columns="exe", values="throughput")
print(df_pivot)


# Reorder columns
exec_order = {
    # name: (order, column_name)
    "ac_cuda": (-0, "ac_cuda"),
    "ac_cuda_fuse_bit": (-1, "ac_cuda_fuse_bit"),
    "ac_cuda_fuse_basic": (1, "Base"),
    "ac_cuda_fuse_adv": (2, "DTM—"),
    "ac_cuda_fuse_all": (3, "DTM"), # "bitGen+$\mathregular{O^1}$"
    "ac_cuda_fuse_all_o2": (-4, "O2"),
    "ac_cuda_fuse_all_o2_1": (-4.1, "O2_1"),
    "ac_cuda_fuse_all_o2_2": (-4.2, "O2_2"),
    "ac_cuda_fuse_all_o2_4": (-4.4, "O2_4"),
    "ac_cuda_fuse_all_o2_8": (4.8, "SR"), # "bitGen+$\mathregular{O^1}$"
    "ac_cuda_fuse_all_o3": (-5, "O3"),
    "ac_cuda_fuse_all_o3_1": (-5.1, "O3_1"),
    "ac_cuda_fuse_all_o3_2": (5.2, "ZBS"),
    "ac_cuda_fuse_all_o3_4": (-5.4, "O3_4"),
    "ac_cuda_fuse_all_o3_8": (-5.8, "O3_8"),
    "ac_torch": (-11, "ac_torch"),
    "ac_torch_fuse": (-12, "ac_torch_fuse"),
    "hs": (-21, "hs"),
    "icgrep": (-22, "icgrep"),
    "ngap": (-31, "ngap"),
}

# Reorder and rename columns based on exec_order
column_reorder = []
for exe in df_pivot.columns.tolist():
    if exe in exec_order and exec_order[exe][0] >= 0:
        column_reorder.append([exe, exec_order[exe][0], exec_order[exe][1]])
column_reorder = sorted(column_reorder, key=lambda x: x[1])
df_pivot = df_pivot[[col[0] for col in column_reorder]]
df_pivot.rename(columns={col[0]: col[2] for col in column_reorder}, inplace=True)

# Reorder and remove rows based on app_order
app_order = {
    "Brill": (0, "Brill"),
    "CAV": (1, "ClamAV"),
    "Dotstar": (2, "Dotstar"),
    "Protomata": (3, "Protomata"),
    "Snort": (4, "Snort"),
    "Yara": (5, "Yara"),
    "Bro217": (11, "Bro217"),
    "ExactMatch": (12, "ExactMatch"),
    "Ranges1": (13, "Ranges1"),
    "TCP": (14, "TCP"),
}
row_reorder = []
for app in df_pivot.index.tolist():
    if app in app_order and app_order[app][0] >= 0:
        row_reorder.append([app, app_order[app][0], app_order[app][1]])
    elif "_8" in app:
        row_reorder.append([app, 998, app])
    elif "_16" in app:
        row_reorder.append([app, 999, app])
row_reorder = sorted(row_reorder, key=lambda x: x[1])
df_pivot = df_pivot.reindex([row[0] for row in row_reorder])
df_pivot.rename(index={row[0]: row[2] for row in row_reorder}, inplace=True)


print("Original data:")
print(df_pivot)

df_pivot = df_pivot.fillna(0.1)

# Normalize the throughput values to the 'ngap' column
df_pivot = df_pivot.div(df_pivot['Base'], axis=0)
# df_pivot = df_pivot.drop(columns=["ngap"], axis=1)
print("Normalized data:")
print(df_pivot)


apps = df_pivot.index.tolist()
exes = df_pivot.columns.tolist()
print("apps", apps)
print("exes", exes)
figure_path = os.path.join(
    os.environ["BITGEN_ROOT"], "results", "figures", "throughput_fullapp_1input_breakdown.pdf"
)
os.makedirs(os.path.dirname(figure_path), exist_ok=True)
figurePlotter.bar(
    apps,
    exes,
    df_pivot.values,
    plotSize=(7, 2),
    filename=figure_path,
    groupsInterval=0.2,
    xyConfig={
        "xylabel": ["", "Throughput\nnormalized to Base"],
        "xlim": [None, None],
        "ylim": [0, 50],
        "labelExceedYlim": True,
        "labelAllY": False,
        "xyscale": [None, None],
        "showxyTicksLabel": [True, True],
        "xyticksRotation": [30, 0],
        "xyticksMajorLocator": [None, None],
    },
    legendConfig={
        "position": "lower center",
        "positionOffset": (0.595, 0.98), # (0.56, 0.95)
        "col": 10,
        "legend.columnspacing": 1,
        "legend.handlelength": 2,
        "legend.handletextpad": 0.8,
    },
)
