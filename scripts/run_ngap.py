import argparse
import os
import re
import sys
from colorama import Fore, Style

from bitgen.tool.timer import global_timer
from bitgen.tool import exe_command
from bitgen.tool.pcre2anml import pcre2anml


def convert_regex_file_to_anml(regex_file, regex_num=-1):
    return pcre2anml(regex_file, None, regex_num=regex_num, random_select=False)


def check_regex_file(file, regex_num=-1):
    if not os.path.isfile(file):
        print(f"Input file '{file}' does not exist.")
        raise FileNotFoundError
    file = convert_regex_file_to_anml(file, regex_num=regex_num)
    assert file.endswith(".anml")
    return file


def gen_command(regex_file, corpus_file, multi_input=1, input_size=-1, precompute_depth=3):
    # O1: nonblockinggroups
    # command = (
    #     f"ngap -a {regex_file} -i {corpus_file} "
    #     "--app-name=regex --algorithm=nonblockingallgroups --input-start-pos=0 "
    #     f"--input-len={input_size} --split-entire-inputstream-to-chunk-size=-1 --group-num=200 "
    #     f"--duplicate-input-stream={multi_input} --unique=false --unique-frequency=10 --use-soa=false "
    #     "--result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=25600000 "
    #     "--add-aan-start=256 --add-aas-interval=32 --active-threshold=10 "
    #     "--precompute-cutoff=-1 --precompute-depth=2 --compress-prec-table=true "
    #     "--report-off=false --validation=true"
    # )

    # OA
    group_num = 256 // multi_input  # 256 for 3090
    if group_num < 1:
        group_num = 1
    chunk_size = -1
    if multi_input > 1:
        chunk_size = input_size
    command = (
        f"ngap -a {regex_file} -i {corpus_file} "
        "--app-name=regex --algorithm=nonblockingallgroups --input-start-pos=0 "
        f"--input-len={input_size} --split-entire-inputstream-to-chunk-size={chunk_size} --group-num={group_num} "
        f"--duplicate-input-stream={multi_input} --unique=false --unique-frequency=10 --use-soa=false "
        "--result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=25600 "
        "--add-aan-start=1000000 --add-aas-interval=256 --active-threshold=16 "
        f"--precompute-cutoff=-1 --precompute-depth={precompute_depth} --compress-prec-table=true "
        "--pc-use-uvm=false --report-off=false --remove-degree=false "
        "--quit-degree=false --max-nfa-size=-1 --adaptive-aas=true "
        # "--use-unique-matchset=true --remove-loop-edge=true --loop-state-prefetch=false "
        "--quick-validation=1 --validation=false "
    )

    # OAe2: nonblockingalle2groups
    # command = (
    #     f"ngap -a {regex_file} -i {corpus_file} "
    #     "--app-name=regex --algorithm=nonblockingalle2groups --input-start-pos=0 "
    #     f"--input-len={input_size} --split-entire-inputstream-to-chunk-size=-1 --group-num=200 "
    #     f"--duplicate-input-stream={multi_input} --unique=false --unique-frequency=10 --use-soa=false "
    #     "--result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=25600 "
    #     "--add-aan-start=256 --add-aas-interval=256 --active-threshold=16 "
    #     "--precompute-cutoff=-1 --precompute-depth=3 --compress-prec-table=true "
    #     "--pc-use-uvm=false --report-off=false --remove-degree=false " "--quit-degree=false --max-nfa-size=-1 --adaptive-aas=true "
    #     "--use-unique-matchset=true --remove-loop-edge=true --loop-state-prefetch=false "
    #     "--quick-validation=1 --validation=false "
    # )

    return command


def run_ngap(cmd):
    result = exe_command(cmd, check=False, shell=False)
    return result


def get_count(output):
    count = re.search(r"Results number:\s*([\d.]+)", output)
    if count:
        count = count.group(1)
    else:
        raise Exception("Failed to get count.")
    return int(count)


def get_throughput(output):
    throughput = re.search(r"throughput\s*=\s*([\d.]+)\s*MB/s", output)
    if throughput:
        throughput = throughput.group(1)
    else:
        raise Exception("Failed to get throughput.")
    return float(throughput)


def get_time(output):
    time = re.search(r"ngap elapsed time:\s*([\d.]+)\s*seconds", output)
    if time:
        time = time.group(1)
    else:
        raise Exception("Failed to get throughput.")
    return float(time) * 1000  # ms


# def exe_ngap_expr(e, c, repeat_num = 1, debug=False):
#     regex_tmp_file = "/tmp/ngap_regex_tmp.regex"
#     regex = e
#     first_slash = regex.find("/")
#     last_slash = regex.rfind("/")
#     if not (first_slash != -1 and last_slash != -1 and first_slash != last_slash):
#         regex = f"/{regex}/"
#     regex = regex.replace("'", "'\\''")
#     regex_tmp_cmd = f"echo '{regex}' > {regex_tmp_file}"
#     exe_command(regex_tmp_cmd, check=True)
#     re_file = check_regex_file(regex_tmp_file)
#     command = gen_command(re_file, c, repeat_num)
#     # for i in range(5):
#     #     result = exe_command(command)
#     result = exe_command(command, check=False, shell=True)
#     if debug:
#         print("result.returncode: ", result.returncode)
#         print("result.stdout: ", result.stdout)
#         print("result.stderr: ", result.stderr)
#     count = get_count(result.stdout)
#     throughput = get_throughput(result.stdout)
#     time = get_time(result.stdout)
#     global_timer.add_timing("ngap", time)

#     input_repeats = 1
#     input_size = os.path.getsize(c) * input_repeats / 1024 / 1024  # MB
#     global_timer.append_data("ngap", "input_size", input_size)

#     avg_duration = global_timer.get_value("ngap", "avg_duration")
#     throughput = input_size / avg_duration * 1024  # MB/s
#     global_timer.append_data("ngap", "throughput", throughput)
#     global_timer.append_data("ngap", "count", count)
#     print(f"Count: {count}")
#     print(f"Throughput: {throughput} MB/s")

#     return count

# def exe_ngap_file_one(f, c, debug=False):
#     with open(f, "r") as file:
#         regex_id = 0
#         count_all = 0
#         for line in file:
#             regex = line.strip()
#             if len(regex) == 0:
#                 continue
#             print(f"{Fore.GREEN}PCRE[{regex_id}]: {regex}{Style.RESET_ALL}")
#             count = exe_ngap_expr(regex, c, debug=debug)
#             count_all += count
#             regex_id += 1
#             print(f"total count until now: {count_all}")
#         print(f"Total count: {count_all}, regex count: {regex_id}")
#     return count_all


def run_regex_file(
    regex_file, input_path, input_size, multi_input=1, regex_num=-1, precompute_depth=3
):
    warmup_iters = 2
    exec_iters = 3
    re_file = check_regex_file(regex_file, regex_num=regex_num)

    cmd = gen_command(
        re_file,
        input_path,
        multi_input=multi_input,
        input_size=input_size,
        precompute_depth=precompute_depth,
    )
    # warmup
    for _ in range(warmup_iters):
        ngap_output = run_ngap(cmd)
    # run
    for _ in range(exec_iters):
        ngap_output = run_ngap(cmd)
        ngap_time = get_time(ngap_output.stdout)
        global_timer.add_timing("run_regex", ngap_time)  # ms
        # print(ngap_output.stdout)

    ngap_count = get_count(ngap_output.stdout)
    input_size = multi_input * input_size / 1024 / 1024  # MB
    global_timer.append_data("run_regex", "input_size", input_size)
    avg_duration = global_timer.get_value("run_regex", "avg_duration")
    throughput = input_size / avg_duration * 1024  # MB/s
    global_timer.append_data("run_regex", "throughput", throughput)
    # global_timer.append_data("run_regex", "count", ngap_count)
    return ngap_count


def main():
    args_parser = argparse.ArgumentParser(description="Benchmark ngAP.")
    # parser.add_argument(
    #     "-e",
    #     type=str,
    #     metavar="STRING",
    #     help="regex.",
    # )
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
        "--precompute-depth",
        default=3,
        type=int,
        help="Precompute depth.",
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
        multi_input=args.multi_input,
        regex_num=args.regex_num_from_file,
        precompute_depth=args.precompute_depth,
    )


if __name__ == "__main__":
    try:
        global_timer.reset_timings()
        main()
    finally:
        global_timer.display_timings()
        global_timer.reset_timings()
