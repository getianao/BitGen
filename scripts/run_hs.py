import sys
import argparse
from colorama import Fore, Style
import re
import os

from bitgen.tool.command_tool import exe_command
from bitgen.tool.timer import global_timer
from bitgen.tool.pcre2hs import pcre2hs

from run_bitgen import select_regex_from_file


def get_hs_count(output):
    count = re.search(r"total_count\s*=\s*(\d+)", output)
    if count:
        count = count.group(1)
    else:
        raise Exception(f"Failed to get hs count.\noutput: {output}")
    return int(count)


def get_hs_time(output):
    time = re.search(r"delta_us\s*=\s*([\d.]+)\s*", output)
    if time:
        time = time.group(1)
    else:
        raise Exception("Failed to get hs time.")
    return float(time) / 1000


def run_hs(file_path, text, multi_input=1, input_size=-1, threads = 1, verbose=True):
    if isinstance(file_path, list):
        hs_cmd = [
            "hsrun",
            "--support",
            "-t",
            f"{threads}",
            "-d",
            f"{multi_input}",
            "-i",
            f"{input_size}",
            *file_path,  # Unpack the list of file paths
            text,
        ]
    else:
        hs_cmd = [
            "hsrun",
            "--support",
            "-t",
            f"{threads}",
            "-d",
            f"{multi_input}",
            "-i",
            f"{input_size}",
            file_path,
            text,
        ]
    if threads >= 1:
        hs_cmd = ["taskset", "-c", f"0-{threads - 1}"] + hs_cmd
    output = exe_command(hs_cmd, check=False, shell=False, verbose=verbose)
    return output


def split_chunks(lst, n):
    """Split list lst into n chunks as evenly as possible."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def run_regex_file(
    regex_file, input_path, input_size, input_repeats=1, multi_input=1, regex_num=-1, threads=1
):
    warmup_iters = 2
    exec_iters = 3
    assert input_repeats == 1
    assert regex_num == -1

    if threads > 1:
        regex_list = select_regex_from_file(
            regex_file,
            regex_num=-1,
            random_select=False,
        )
        regex_chunks = split_chunks(regex_list, threads)

        hs_file = []
        for i, chunk in enumerate(regex_chunks):
            if not chunk:
                print(f"Warning: regex chunk {i} is empty, skipping.")
                continue
            file_path = pcre2hs(chunk, suffix=f"{os.path.basename(regex_file)}_{i}")
            hs_file.append(file_path)

        if not hs_file:
            raise RuntimeError("All regex chunks are empty. No regex to process.")

        print(f"Running with {threads} threads, regex files: {hs_file}")
    else:
        hs_file = pcre2hs(regex_file, suffix=os.path.basename(regex_file))

    # warmup
    for _ in range(warmup_iters):
        hs_output = run_hs(
            hs_file,
            input_path,
            multi_input=multi_input,
            input_size=input_size,
            verbose=True,
            threads=threads,
        )

    # run
    for _ in range(exec_iters):
        hs_output = run_hs(
            hs_file,
            input_path,
            multi_input=multi_input,
            input_size=input_size,
            verbose=False,
            threads=threads,
        )
        # print(hs_output.stdout)
        hs_time = get_hs_time(hs_output.stdout)
        hs_count = get_hs_count(hs_output.stdout)
        global_timer.add_timing("run_regex", hs_time)
        # print(hs_output.stdout)
    input_size_mb = multi_input * input_repeats * input_size / 1024 / 1024
    global_timer.append_data("run_regex", "input_size", input_size_mb)
    avg_duration = global_timer.get_value("run_regex", "avg_duration")
    throughput = input_size_mb / avg_duration * 1024  # MB/s
    global_timer.append_data("run_regex", "throughput", throughput)
    # global_timer.append_data("run_regex", "count", hs_count)
    print(f"hs count: {hs_count}")
    return hs_count


def main():
    args_parser = argparse.ArgumentParser(description="Benchmarking hs")
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
    
    args_parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Number of threads to use. Parallel on regexes.",
    )
    

    args = args_parser.parse_args()
    run_regex_file(
        args.regex_file,
        args.input_file,
        args.input_size,
        multi_input=args.multi_input,
        regex_num=args.regex_num_from_file,
        threads=args.threads,
    )


if __name__ == "__main__":
    try:
        global_timer.reset_timings()
        main()
    finally:
        global_timer.display_timings()
        global_timer.reset_timings()
