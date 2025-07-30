import sys
import argparse
import os
import logging
from colorama import Fore, Style

from bitgen.tool.pcre2pablo import run_icgrep_from_file as run
from bitgen.tool.pcre import pcre2file
from bitgen.from_icgrep import (
    get_icgrep_count,
    get_throughput,
    get_icgrep_time,
)
from bitgen.tool.timer import global_timer


def run_regex_file(
    regex_file, input_path, input_size, input_repeats=1, multi_input=1, regex_num=-1
):
    warmup_iters = 2
    exec_iters = 3
    rf_path = os.path.join("/tmp/", os.path.basename(regex_file) + ".regex")
    regex_file = pcre2file(
        regex_file, rf_path, regex_num=regex_num, random_select=False
    )
    
    if "snort.1chip.fix.icgrep.anml.unique.regex" in regex_file:
        # T00 slow
        warmup_iters = 1
        exec_iters = 1
    # warmup
    for _ in range(warmup_iters):
        for input_id in range(multi_input):
            icgrep_output = run(
                regex_file,
                input_path,
                repeat=input_repeats,
                input_size=input_size,
                verbose=True,
            )
    # run
    for _ in range(exec_iters):
        icgrep_time_multi_input = 0
        icgrep_count = 0
        for input_id in range(multi_input):
            icgrep_output = run(
                regex_file,
                input_path,
                repeat=input_repeats,
                input_size=input_size,
                verbose=False,
            )
            icgrep_time_multi_input += get_icgrep_time(icgrep_output.stdout)
            icgrep_count += get_icgrep_count(icgrep_output.stdout)
        global_timer.add_timing("run_regex", icgrep_time_multi_input)
        # print(icgrep_output.stdout)
    input_size = multi_input * input_repeats * input_size / 1024 / 1024  # MB
    global_timer.append_data("run_regex", "input_size", input_size)
    avg_duration = global_timer.get_value("run_regex", "avg_duration")
    throughput = input_size / avg_duration * 1024  # MB/s
    global_timer.append_data("run_regex", "throughput", throughput)
    # global_timer.append_data("run_regex", "count", icgrep_count)
    print(f"icgrep count: {icgrep_count}")
    return icgrep_count


def main():
    args_parser = argparse.ArgumentParser(description="Benchmarking icgrep")
    args_parser.add_argument(
        "-f",
        "--regex_file",
        default="",
        type=str,
        required=True,
        help="Regex file path.",
    )
    args_parser.add_argument(
        "-i",
        "--input_file",
        default="",
        type=str,
        required=True,
        help="Input file path.",
    )
    args_parser.add_argument(
        "--input-size",
        default=1000000,
        type=int,
        help="Input size in Byte.",
    )
    args_parser.add_argument(
        "--multi-input",
        default=1,
        type=int,
        help="Number of input stream.",
    )
    args_parser.add_argument(
        "--regex_num_from_file",
        default=-1,
        type=int,
        help="Number of regexes. Only used for regex file.",
    )

    args = args_parser.parse_args()
    run_regex_file(
        args.regex_file,
        args.input_file,
        args.input_size,
        multi_input=1,
        input_repeats=args.multi_input,  # Use input_repeats to simulate multi_input
        regex_num=args.regex_num_from_file,
    )


if __name__ == "__main__":
    try:
        global_timer.reset_timings()
        main()
    finally:
        global_timer.display_timings()
        global_timer.reset_timings()
