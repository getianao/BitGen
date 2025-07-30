import argparse
import yaml
import pandas as pd
import os
from figurePlotter.df2latex import LatexPrinter

import re
import math

def analyze_regex(pattern):
    # 结果字典
    result = {
        "concatenation": 0,
        "alternation": 0,
        "kleene_star": 0,
        "bounded_repetition": 0,
        "one_or_more": 0,
        "zero_or_one": 0
    }
    
    # 去掉首尾的 /（如果有）
    if pattern.startswith('/') and pattern.endswith('/'):
        pattern = pattern[1:-1]
    
    # 计算 alternation (|)
    result["alternation"] = pattern.count('|')
    
    # 计算 Kleene star (*)
    result["kleene_star"] = sum(1 for i, c in enumerate(pattern) if c == '*' and (i == 0 or pattern[i-1] != '\\'))
    
    # 计算 one or more (+)
    result["one_or_more"] = sum(1 for i, c in enumerate(pattern) if c == '+' and (i == 0 or pattern[i-1] != '\\'))
    
    # 计算 zero or one (?)
    result["zero_or_one"] = sum(1 for i, c in enumerate(pattern) if c == '?' and (i == 0 or pattern[i-1] != '\\'))
    
    # 计算 bounded repetition ({n}, {n,m})
    bounded_matches = len(re.findall(r'(?<!\\)\{\d+(?:,\d*)?\}', pattern))
    result["bounded_repetition"] = bounded_matches
    
    # 计算 concatenation
    # 将转义序列视为单个单元
    i = 0
    units = []
    while i < len(pattern):
        if pattern[i] == '\\' and i + 1 < len(pattern):
            if pattern[i+1] in 'x':  # \xNN
                if i + 3 < len(pattern):
                    units.append(pattern[i:i+4])
                    i += 4
                else:
                    units.append(pattern[i])
                    i += 1
            else:  # \s, \d 等
                units.append(pattern[i:i+2])
                i += 2
        elif pattern[i] in '|*+?{}':  # 跳过操作符
            i += 1
        else:
            units.append(pattern[i])
            i += 1
    result["concatenation"] = len(units) - 1 if len(units) > 1 else 0
    
    return result

def analyze_regex_file(file_path):
    BITGEN_ROOT = os.environ.get("BITGEN_ROOT", "")
    file_path = file_path.replace("${BITGEN_ROOT}", BITGEN_ROOT)
    result = {
        "concatenation": 0,
        "alternation": 0,
        "kleene_star": 0,
        "bounded_repetition": 0,
        "one_or_more": 0,
        "zero_or_one": 0
    }

    with open(file_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) == 0:
                continue
            result_line = analyze_regex(line)
            result["concatenation"] += result_line["concatenation"]
            result["alternation"] += result_line["alternation"]
            result["kleene_star"] += result_line["kleene_star"]
            result["bounded_repetition"] += result_line["bounded_repetition"]
            result["one_or_more"] += result_line["one_or_more"]
            result["zero_or_one"] += result_line["zero_or_one"]
            
    return result


def file_line_num(file_path):
    BITGEN_ROOT = os.environ.get("BITGEN_ROOT", "")
    file_path = file_path.replace("${BITGEN_ROOT}", BITGEN_ROOT)

    with open(file_path, "r") as f:
        lines = f.readlines()
        line_num = 0
        for i, line in enumerate(lines):
            lines[i] = line.strip()
            if len(lines[i]) > 0:
                line_num += 1
        return line_num

def regex_length_avg_std(file_path):
    BITGEN_ROOT = os.environ.get("BITGEN_ROOT", "")
    file_path = file_path.replace("${BITGEN_ROOT}", BITGEN_ROOT)

    with open(file_path, "r") as f:
        lines = f.readlines()
        line_num = 0
        regex_lengths = []
        for i, line in enumerate(lines):
            first_slash = line.find("/")
            last_slash = line.rfind("/")
            if (
                first_slash == 0
                and first_slash != last_slash
                and last_slash > first_slash
            ):
                # It has slashes
                line = line[first_slash + 1 : last_slash]
            else:
                raise ValueError(f"Regex requires slashes: {line}")
            line = line.strip()
            if len(line) == 0:
                continue
            line_num += 1
            regex_lengths.append(len(line))
        
        if line_num == 0:
            raise ValueError("No valid regex lines found in the file.")
        
        avg_length = sum(regex_lengths) / line_num
        variance = sum((x - avg_length) ** 2 for x in regex_lengths) / line_num
        std_dev = math.sqrt(variance)
        
    return avg_length, std_dev


app_dict = {
    "Brill": {
        "id": 1,
        "suite": "AutomataZoo",
        "full_name": "Brill",
        "Abbr": "Brill",
    },
    "CAV": {
        "id": 2,
        "suite": "AutomataZoo",
        "full_name": "ClamAV",
        "Abbr": "CAV",
    },
    "Dotstar": {
        "id": 3,
        "suite": "AutomataZoo",
        "full_name": "Dotstar",
        "Abbr": "DS",
    },
    "Protomata": {
        "id": 4,
        "suite": "AutomataZoo",
        "full_name": "Protomata",
        "Abbr": "Pro",
    },
    "Snort": {
        "id": 5,
        "suite": "AutomataZoo",
        "full_name": "Snort",
        "Abbr": "Snort",
    },
    "Yara": {
        "id": 6,
        "suite": "AutomataZoo",
        "full_name": "Yara",
        "Abbr": "Yara",
    },
    "Bro217": {
        "id": 7,
        "suite": "Regex",
        "full_name": "Bro217",
        "Abbr": "Bro",
    },
    "ExactMatch": {
        "id": 8,
        "suite": "Regex",
        "full_name": "ExactMatch",
        "Abbr": "EM",
    },
    "Ranges1": {
        "id": 9,
        "suite": "Regex",
        "full_name": "Ranges1",
        "Abbr": "Ran1",
    },
    "TCP": {
        "id": 10,
        "suite": "Regex",
        "full_name": "TCP",
        "Abbr": "TCP",
    },
}

app_config_path = "/home/tge/workspace/automata-compiler/configs/app/full/app_full.yaml"
data = []

with open(app_config_path, "r") as f:
    app_config = yaml.safe_load(f)

root = app_config["root"]
for app in app_config["apps"]:
    print(app["name"])
    filtered_regex = root + app["regex"]
    original_regex = filtered_regex.replace("icgrep.anml.unique.", "")
    icgrep_regex = filtered_regex.replace("icgrep.anml.unique.", "icgrep.")
    repeat_regex = filtered_regex.replace("icgrep.anml.unique.", "icgrep.anml.")

    original_regex_num = file_line_num(original_regex)
    icgrep_regex_num = file_line_num(icgrep_regex)
    repeat_regex_num = file_line_num(repeat_regex)
    filtered_regex_num = file_line_num(filtered_regex)
    regex_length, regex_length_std = regex_length_avg_std(filtered_regex)
    # regex_attr = analyze_regex_file(filtered_regex)
    data.append(
        [
            app["name"],
            original_regex_num,
            original_regex_num - icgrep_regex_num,
            icgrep_regex_num - repeat_regex_num,
            repeat_regex_num - filtered_regex_num,
            "",
            app_dict[app["name"]]["full_name"],
            app_dict[app["name"]]["Abbr"],
            filtered_regex_num,
            regex_length,
            regex_length_std,
            # regex_attr["concatenation"],
            # regex_attr["alternation"],
            # regex_attr["kleene_star"],
            # regex_attr["bounded_repetition"],
            # regex_attr["one_or_more"],
            # regex_attr["zero_or_one"],
        ]
    )

df_full = pd.DataFrame(
    data,
    columns=[
        "app",
        "original",
        "icgrep",
        "pcre2mnrl + vasim",
        "repeat",
        "Suite",
        "Application",
        "Abbr.",
        "#Regex",
        "Avg Regex Length",
        "Std Regex Length",
        # "Concatenation",
        # "Alternation",
        # "Kleene Star",
        # "Bounded Repetition",
        # "One or More",
        # "Zero or One",
    ],
)
print(df_full)

df = df_full[["Application", "#Regex"]]  # "Avg Regex Length" "Suite", "Abbr.", 
df = df.set_index("Application").T
print(df)


def print_latex_table(df, table_path):
    lp = LatexPrinter(df)
    lp.bold_max = False
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    lp.gen_table_latex(table_path)


print_latex_table(
    df, os.path.join(os.environ["BITGEN_ROOT"], "results", "tables", "app_info.tex")
)

print_latex_table(
    df_full,
    os.path.join(os.environ["BITGEN_ROOT"], "results", "tables", "app_info_full.tex"),
)
