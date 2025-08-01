import os
import pandas as pd
import sys
from scipy.stats import gmean
import figurePlotter
from figurePlotter.df2latex import LatexPrinter

from process_result import process_two_stage


results_baseline_cpu_csv = os.path.join(
    os.environ["BITGEN_ROOT"], "results/csv/exec_baseline_arpg_1input.csv"
)
results_baseline_cpu_hs_csv = os.path.join(
    os.environ["BITGEN_ROOT"], "results/csv/exec_baseline_hs_arpg_1input.csv"
)

results_csv = os.path.join(
    os.environ["BITGEN_ROOT"], "results/csv/exec_tune_best_arpg_1input.csv"
)
output_name = "throughput_fullapp_1input_new2"

def load_csv(baseline_path, results_path):
    df = pd.read_csv(results_path)
    df = process_two_stage(df=df)
    df2 = pd.read_csv(baseline_path)

    df = pd.concat([df, df2], ignore_index=True)  # Resets index after merging
    df = df[df["name"] == "run_regex"]
    df = df[["exe", "app", "throughput"]]
    return df

df_cpu = pd.read_csv(results_baseline_cpu_csv)
df_cpu = df_cpu[["exe", "app", "throughput"]]
df_cpu_hs = pd.read_csv(results_baseline_cpu_hs_csv)
df_cpu_hs = df_cpu_hs.loc[df_cpu_hs.groupby(["app"])["throughput"].idxmax()]
df_cpu_hs["exe"] = "hs"
# print("df_cpu_hs", df_cpu_hs)
df_cpu_hs = df_cpu_hs[["exe", "app", "throughput"]]
df_cpu_hs_1core = pd.read_csv(results_baseline_cpu_hs_csv)
df_cpu_hs_1core = df_cpu_hs_1core[df_cpu_hs_1core["exe"] == "hs"]
df_cpu_hs_1core["exe"] = "hs_1"
df = pd.read_csv(results_csv)
df = process_two_stage(df=df)
df = df[df["name"] == "run_regex"]
df = df[["exe", "app", "throughput"]]

df = pd.concat([df, df_cpu, df_cpu_hs, df_cpu_hs_1core], ignore_index=True)

print(df)

# df = pd.concat([df, df_8input, df_16input], ignore_index=True)
df_pivot = df.pivot(index="app", columns="exe", values="throughput")
print(df_pivot)


# Reorder columns
exec_order = {
    # name: (order, column_name)
    "ac_cuda": (-0, "ac_cuda"),
    "ac_cuda_fuse_bit": (-1, "ac_cuda_fuse_bit"),
    "ac_cuda_fuse_adv": (-2, "ac_cuda_fuse_adv"),
    "ac_cuda_fuse_all": (-3, "ac"),
    "ac_cuda_fuse_all_o2": (-4, "ac_o2"),
    "ac_cuda_fuse_all_o3": (5, "bitGen"),
    "ac_torch": (-11, "ac_torch"),
    "ac_torch_fuse": (-12, "ac_torch_fuse"),
    "hs_1": (20.5, "HS-1T"),
    "hs": (21, "HS-MT"),
    "hs_2": (22, "Hyperscan_2"),
    "hs_4": (23, "Hyperscan_4"),
    "hs_8": (24, "Hyperscan_8"),
    "hs_16": (25, "Hyperscan_16"),
    "hs_32": (26, "Hyperscan_32"),
    "icgrep": (42, "icgrep"),
    "ngap": (31, "ngAP"),
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

apps = df_pivot.index.tolist()
exes = df_pivot.columns.tolist()
print("apps", apps)
print("exes", exes)

print("Original data:")
print(df_pivot)

df_pivot = df_pivot.fillna(0.1)


# Normalize the throughput values to the 'ngap' column
df_pivot_norm = df_pivot.div(df_pivot['ngAP'], axis=0)
print("Normalized data:")
print(df_pivot_norm)

figure_path = os.path.join(
    os.environ["BITGEN_ROOT"], "results", "figures", f"{output_name}.pdf"
)
os.makedirs(os.path.dirname(figure_path), exist_ok=True)
figurePlotter.bar(
    apps,
    exes,
    df_pivot_norm.values,
    plotSize=(7, 2),
    filename=figure_path,
    groupsInterval=0.2,
    xyConfig={
        "xylabel": ["", "Throughput\nnormalized to ngAP"],
        "xlim": [None, None],
        "ylim": [0.1, 1000],
        "labelExceedYlim": True,
        "labelAllY": False,
        "xyscale": [None, 'log'],
        "showxyTicksLabel": [True, True],
        "xyticksRotation": [30, 0],
        "xyticksMajorLocator": [None, None],
    },
    legendConfig={
        "position": "lower center",
        "positionOffset": (0.5, 0.95), # (0.5, 0.5),
        "col": 10,
        "legend.columnspacing": 1,
        "legend.handlelength": 2,
        "legend.handletextpad": 0.8,
    },
)


df_latex = df_pivot.round(1)
df_latex.reset_index(inplace=True)

exes = df_latex.columns.tolist()
exes = [exe for exe in exes if exe != "app"]
print("exes", exes)
exe_norm = "bitGen"

data = []
for row in df_latex.iterrows():
    max_exe = row[1][exes].idxmax()
    data_row = {"app": row[1]["app"]}
    for exe in exes:
        if exe not in row[1]:
            raise ValueError(f"Missing {exe} in {row[1]}")
        norm = row[1][exe_norm] / row[1][exe]
        output = row[1][exe]
        if exe == max_exe:
            output = f"\\textbf{{{output:.1f}}}"
        else:
            output = f"{output}"
        if exe == exe_norm:
            output = f"{output}"
        else:
            output = f"{output} & {norm:.1f}$\\times$"
        
        df_latex.at[row[0], exe] = output
        data_row[exe] = row[1][exe]
        data_row[exe+"_norm"] = row[1][exe_norm] / row[1][exe]
    data.append(data_row)
df_new = pd.DataFrame(data)

gmean_row = {"app": "gmean"}
for exe in exes:
    gmean_row[exe] = gmean([val for val in df_new[exe] if isinstance(val, (int, float)) and val > 0])
    gmean_row[exe + "_norm"] = gmean([val for val in df_new[exe + "_norm"] if isinstance(val, (int, float)) and val > 0])

df_new = pd.concat([df_new, pd.DataFrame([gmean_row])], ignore_index=True)
print(df_new)
df_new_path = os.path.join(
    os.environ["BITGEN_ROOT"], "results", "tables", f"{output_name}.csv"
)
os.makedirs(os.path.dirname(df_new_path), exist_ok=True)
df_new.to_csv(
    df_new_path,
    index=False,
)
print("Table saved to CSV:", df_new_path)


# print("df_latex", df_latex)
# lp = LatexPrinter(df_latex)
# lp.bold_max = False
# lp.tex_table_header_position = r"{|c|r|r|r|r|}"
# table_path = os.path.join(
#     os.environ["BITGEN_ROOT"], "results", "tables", f"{output_name}.tex"
# )
# os.makedirs(os.path.dirname(table_path), exist_ok=True)
# lp.gen_table_latex(table_path)
