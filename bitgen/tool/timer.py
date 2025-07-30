import time
from contextlib import contextmanager
import pandas as pd
from datetime import datetime
import os
import torch


class Timer:
    def __init__(self):
        self.timing_data = []
        self.group = None
        self.exe = None
        self.app = None
        self.params = None

    class TimerContextManager:
        def __init__(self, store, name="unnamed"):
            self.store = store
            self.name = name

        def __enter__(self):
            torch.cuda.synchronize()
            self.start_time = time.perf_counter() * 1000
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            torch.cuda.synchronize()
            self.end_time = time.perf_counter() * 1000
            self.duration = self.end_time - self.start_time
            self.store.add_timing(self.name, self.duration)

    def set_group(self, group_name):
        self.group = group_name

    def set_exe(self, exe):
        self.exe = exe

    def set_app(self, app):
        self.app = app

    def set_params(self, params):
        self.params = params

    def time(self, name="unnamed"):
        return Timer.TimerContextManager(self, name)

    # Use either `global_timeradd_timing` or `with global_timer.time(""):`
    
    def check_params(self, entry, params):
        if not params:
            return True
        for param_name, param_value in zip(*params):
            if entry[param_name] != param_value:
                return False
        return True
        
    def add_timing(self, name, duration):
        print(f"Timer: {name} took {duration:.4f} ms")
        for entry in self.timing_data:
            if (
                entry["name"] == name
                and entry["group"] == self.group
                and entry["exe"] == self.exe
                and entry["app"] == self.app
                and self.check_params(entry, self.params)
            ):
                entry["duration"] += duration
                entry["exec_num"] += 1
                entry["avg_duration"] = entry["duration"] / entry["exec_num"]
                return
        # If the entry does not exist, create a new one
        new_entry = {
            "group": self.group,
            "exe": self.exe,
            "app": self.app,
            "name": name,
            "exec_num": 1,
            "duration": duration,
            "avg_duration": duration,
        }
        if self.params:
            for param_name, param_value in zip(*self.params):
                new_entry[param_name] = param_value
        self.timing_data.append(new_entry)

    def append_data(self, name, data_name, data_value):
        for entry in self.timing_data:
            if (
                entry["name"] == name
                and entry["group"] == self.group
                and entry["exe"] == self.exe
                and entry["app"] == self.app
                and self.check_params(entry, self.params)
            ):
                entry[data_name] = data_value
                return
        raise ValueError(f"Entry {name} not found")

    def get_value(self, name, data_name="duration"):
        for entry in self.timing_data:
            if (
                entry["name"] == name
                and entry["group"] == self.group
                and entry["exe"] == self.exe
                and entry["app"] == self.app
                and self.check_params(entry, self.params)
            ):
                return entry[data_name]
        raise ValueError(f"Entry {name} not found")

    def add_row(self, row):
        self.timing_data.append(row)

    def display_timings(self):
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", 40)
        pd.set_option("display.width", 1000)
        if not self.timing_data:
            print("No timing data collected.")
            return

        print("\n----- Timing Data -----")
        df = pd.DataFrame(self.timing_data)
        print(df)

        if "count" in df.columns and "ref_count" in df.columns:
            total_count = df["count"].sum()
            total_count_ref = df["ref_count"].sum()
            print(f"Total count: {total_count}")
            print(f"Total count ref: {total_count_ref}")
            true_count = (
                df["check"]
                .apply(lambda x: True if x == True or x == "True" else False)
                .sum()
            )
            false_count = (
                df["check"]
                .apply(lambda x: True if x == False or x == "False" else False)
                .sum()
            )
            print(f"True count: {true_count}")
            print(f"False count: {false_count}")
        print("\n-----------------------\n")

    def analyze_average_duration(self):
        if not self.timing_data:
            print("No timing data to analyze.")
            return
        total_duration = sum(entry["duration"] for entry in self.timing_data)
        average_duration = total_duration / len(self.timing_data)
        print("----- Average Execution Time -----\n")
        print(
            f"Average Duration: {average_duration:.6f} seconds over {len(self.timing_data)} run(s)"
        )
        print("\n----------------------------------\n")

    def get_data(self):
        return self.timing_data

    def get_pd_data(self):
        return pd.DataFrame(self.timing_data)

    def save_to_file(self, file_path, df=None):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if df is None:
            df = pd.DataFrame(self.timing_data)
        formatted_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(file_path, f"{formatted_time}.csv")
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def reset_timings(self):
        self.timing_data = []


global_timer = Timer()
