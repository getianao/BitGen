import pandas as pd
import argparse


def process_two_stage(df=None, file=None):
    if df is None:
        if file is None:
            raise ValueError("Either df or file must be provided")
        df = pd.read_csv(file)
    execs = df["exe"].unique()
    apps = df["app"].unique()
    name = df["name"].unique()
    print(f"execs: {execs}")
    print(f"apps: {apps}")
    new_df = pd.DataFrame()
    for app in apps:
        for exec in execs:
            rows = df[(df["app"] == app) & (df["exe"] == exec)]
            # print(f"app: {app}, exec: {exec}")
            # print(rows)
            if rows.empty or not len(rows["name"].unique()) == len(name):
                print(f"Warning: Skipping {app}, {exec} due to insufficient data")
                continue
            row = rows[rows["name"] == "run_regex"].copy()
            row["duration"] = rows["duration"].sum()
            row["avg_duration"] = row["duration"] / row["exec_num"]
            assert (
                rows["input_size"].nunique() == 1
            ), "input_size values are not the same"
            row["throughput"] = row["input_size"] / row["avg_duration"] * 1000  # MB/s
            new_df = pd.concat([new_df, row])
            # print(row)
    # print(new_df)
    return new_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two-stage results")
    parser.add_argument("file", type=str, help="Path to the CSV file")
    args = parser.parse_args()
    file_path = args.file
    # file_path = "/home/tge/workspace/automata-compiler/raw_results/ac/20250224-112040.csv"
    df = process_two_stage(file=file_path)
    print(df)
