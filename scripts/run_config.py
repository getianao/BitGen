import argparse
import yaml
import subprocess
import sys
import os
import pandas as pd
import io
from itertools import product

from bitgen.tool.timer import global_timer
from bitgen.tool.command_tool import exe_command

from run_bitgen import main as main_ac
from run_icgrep import main as main_icgrep
from run_hs import main as main_hs
from run_ngap import main as main_ngap

from plot.process_result import process_two_stage


exec_input_flags = {
    "input_regex_file_flag": "f",
    "input_regex_str_flag": "e",
    "input_stream_flag": "i",
}

def gen_flag_string(name, value):
    if value is None:
        return ""
    if len(name) == 1:
        return f"-{name}={value}"
    elif len(name) == 0:
        return f"{value}"
    else:
        return f"--{name}={value}"


def gen_cmd(app_config, exec_config, options):
    cmds = []
    options_list = []
    if len(options) > 2:
        options = options[1:-1] # remove ""
        options_list = options.split(",")
        print(f"Adding options: {options_list}")
    for app in app_config["apps"]:
        for exec in exec_config["schemes"]:
            cmd = [exec["env"]]
            # Set input regex file
            if app["type"] == "str":
                # Write all str regexes to a file
                regex = app["regex"]
                if not isinstance(regex, list):
                    regex = [regex]
                tmp_regex_file = f"/tmp/regex_str_{app['name']}.txt"
                with open(tmp_regex_file, "w") as f:
                    for r_id, r in enumerate(regex):
                        if r_id != 0:
                            f.write("\n")
                        r = "/" + r + "/"
                        f.write(r)
                cmd.append(
                    gen_flag_string(
                        exec_input_flags["input_regex_file_flag"], tmp_regex_file
                    )
                )
            elif app["type"] == "xml":
                cmd.append("--xml")
            else:
                regex_path = os.path.join(app_config["root"], app["regex"])
                cmd.append(
                    gen_flag_string(
                        exec_input_flags["input_regex_file_flag"], regex_path
                    )
                )
            # Set input stream file
            input = os.path.join(app_config["root"], app["input"])
            cmd.append(gen_flag_string(exec_input_flags["input_stream_flag"], input))

            # Set parameters
            param_value_list = []
            for param_name, param_value in exec_config["common_params"].items():
                if isinstance(param_value, list):
                    param_value_list.append((param_name, param_value))
                else:
                    cmd.append(gen_flag_string(param_name, param_value))
            if exec["params"] is not None:
                for param_name, param_value in exec["params"].items():
                    if isinstance(param_value, list):
                        param_value_list.append((param_name, param_value))
                    else:
                        cmd.append(gen_flag_string(param_name, param_value))
            if app.get("params") is not None:
                for param_name, param_value in app["params"].items():
                    if isinstance(param_value, list):
                        param_value_list.append((param_name, param_value))
                    else:
                        cmd.append(gen_flag_string(param_name, param_value))

            # Generate combinations of parameter values in list
            if len(param_value_list) > 0:
                for param_values in product(*[pv[1] for pv in param_value_list]):
                    exec_params_name = exec["name"]
                    cmd_params = cmd.copy()
                    for v_id, v in enumerate(param_values):
                        exec_params_name += f"_{v}"
                        cmd_params.append(gen_flag_string(param_value_list[v_id][0], v))
                    cmd_params = process_cmd(app["name"], exec_params_name, exec["type"], cmd_params)
                    if len(options_list) > 0:
                        for option in options_list:
                            cmd_params.append(option)
                    if (cmd_params is not None) and (
                        not "app" in exec or app["name"] in exec["app"]
                    ):
                        cmds.append(
                            {
                                "app": app["name"],
                                "exec": exec_params_name,
                                "type": exec["type"],
                                "cmd": cmd_params,
                                "params": ([pv[0] for pv in param_value_list], param_values),
                            }
                        )
            else:
                cmd = process_cmd(app["name"], exec["name"], exec["type"], cmd)
                if len(options_list) > 0:
                    for option in options_list:
                        cmd.append(option)
                if (cmd is not None) and (
                    not "app" in exec or app["name"] in exec["app"]
                ):
                    cmds.append(
                        {
                            "app": app["name"],
                            "exec": exec["name"],
                            "type": exec["type"],
                            "cmd": cmd,
                        }
                    )
    return cmds


def process_cmd(app, exec, type, cmd: list):
    # Environment variable in command
    BITGEN_ROOT = os.environ.get("BITGEN_ROOT", "")
    if BITGEN_ROOT:
        cmd = [arg.replace("${BITGEN_ROOT}", BITGEN_ROOT) for arg in cmd]
    else:
        raise EnvironmentError("BITGEN_ROOT environment variable not set")
    # Special case in configs
    # if app == "Snort" and exec == "icgrep":
    #     return None  # Too slow. ,icgrep,Snort,run_regex,1,144.743,144.743,0.95367431640625,5.72205,476413
    if app == "Snort" and exec.startswith("ac"):
        cmd.append("--check=0")
    if app == "Protomata" and exec == "ngap":
        cmd.append("--precompute-depth=2")
    return cmd

metrics = [
    "sm__inst_executed.sum",
    "sm__inst_executed_pipe_alu.sum",  # Arithmetic Logic Unit
    "sm__inst_executed_pipe_cbu.sum",  # Convergence Barrier Unit.
    "sm__inst_executed_pipe_cbu_pred_off_all.sum",
    "sm__inst_executed_pipe_cbu_pred_on_any.sum",
    # "sm__inst_executed_pipe_xu.sum",# Transcendental and Data Type Conversion Unit
    "sm__inst_executed_pipe_lsu.sum",  # Load/Store Unit
    # "sm__inst_executed_pipe_fp32.sum",
    # "sass__inst_executed_global_loads",
    # "sass__inst_executed_global_stores",
    "sm__throughput.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "smsp__warps_issue_stalled_barrier.sum",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_dispatch_per_warp_active.pct",  # SM dispatch stall
    "l1tex__t_sector_hit_rate.pct",  # L1/TEX hit rate
    "lts__t_sector_hit_rate.pct",  # L2 hit rate
    "smsp__thread_inst_executed_per_inst_executed.pct",  # of active threads per instruction executed
    "dram__bytes.sum",  # of bytes accessed in DRAM
    "dram__bytes_read.sum",  # of bytes read from DRAM
    "dram__bytes_write.sum",  # of bytes written to DRAM
    "sm__sass_data_bytes_mem_shared.sum",
    # "sm__sass_inst_executed_op_shared.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",  # L1/TEX shared memory
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_read.sum",  # L1/TEX shared memory read
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_write.sum",  # L1/TEX shared memory write
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",  # L1/TEX shared memory bank conflicts
]

def cuda_profile():
    kernel_name = "KernelGenerated"
    metrics_str = ",".join(metrics)
    profile_cmd = [
        "ncu",
        "--target-processes",
        "application-only",
        "--metrics",
        metrics_str,
        "--kernel-name",
        kernel_name,
        "--launch-skip",  # skip times
        "5",
        "--launch-count",  # profile times after skipping
        "1",
        "--csv",
        # "--section MemoryWorkloadAnalysis_Chart",
        # "--set base" # full
    ]
    return profile_cmd


def parse_ncu_csv(output_str: str) -> pd.DataFrame:

    csv_start = output_str.find('"ID","Process ID"')
    if csv_start == -1:
        raise ValueError("CSV header not found in ncu output.")

    csv_data = output_str[csv_start:]
    df = pd.read_csv(io.StringIO(csv_data))
    if "Metric Value" in df.columns:
        df["Metric Value"] = df["Metric Value"].astype(str).str.replace(",", "")
        df["Metric Value"] = pd.to_numeric(df["Metric Value"], errors="coerce")
    return df

def call_run(app, exec, type, cmd):
    sys.argv = cmd
    if type == "ac":
        main_ac()
    elif type == "icgrep":
        main_icgrep()
    elif type == "hs":
        main_hs()
    elif type == "ngap":
        main_ngap()
    else:
        raise ValueError(f"Unknown type {type}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--app", type=str, required=True)
    args.add_argument("--exec", type=str, required=True)
    args.add_argument("-d", "--dry-run", default=False, action="store_true")
    args.add_argument("--profile", default=False, action="store_true")

    args.add_argument("--options", type=str, default="")
    args = args.parse_args()

    with open(args.app, "r") as f:
        app_config = yaml.safe_load(f)
    with open(args.exec, "r") as f:
        exec_config = yaml.safe_load(f)
    print("Apps:", [app["name"] for app in app_config["apps"]])
    print("Execs:", [scheme["name"] for scheme in exec_config["schemes"]])

    cmds = gen_cmd(app_config, exec_config, args.options)
    # print(cmds)
    print(f"Total commands: {len(cmds)}")

    try:
        global_timer.reset_timings()
        for cmd in cmds:
            app = cmd["app"]
            exec = cmd["exec"]
            type = cmd["type"]
            command = cmd["cmd"]
            if "params" in cmd:
                print(f"\033[96mAPP: {app}, EXEC: {exec}, PARAMS: {cmd['params']}\033[0m")
            else:
                print(f"\033[96mAPP: {app}, EXEC: {exec}\033[0m")
            if args.profile:
                command = cuda_profile() + command
            print(f"Command: {' '.join(command)}")
            if not args.dry_run:
                if "params" in cmd:
                    global_timer.set_params(cmd["params"])
                global_timer.set_app(cmd["app"])
                global_timer.set_exe(cmd["exec"])

                call_run(app, exec, type, command)
                global_timer.display_timings()

                # if args.profile:
                #     output = exe_command(command)
                #     print("stdout", output.stdout)
                #     # print("stderr", output.stderr)
                #     ncu_output = parse_ncu_csv(output.stdout)
                #     print(ncu_output)
                #     profile_result = {
                #         "app": app,
                #         "exec": exec,
                #     }
                #     for metric in metrics:
                #         filtered = ncu_output[
                #             (ncu_output["Metric Name"] == metric)
                #             & (ncu_output["Kernel Name"] == "KernelGenerated")
                #         ]
                #         if filtered.empty:
                #             raise ValueError(
                #                 f"Metric '{metric}' not found for KernelGenerated"
                #             )
                #         metric_value = filtered["Metric Value"].values[0]
                #         profile_result[metric] = metric_value
                #     print(profile_result)
                #     if not hasattr(global_timer, "profile_data") or global_timer.profile_data is None:
                #         global_timer.profile_data = []
                #     global_timer.profile_data.append(profile_result)
                # else:
                #     call_run(app, exec, type, command)
                #     global_timer.display_timings()

                # # Inst stats
                # if hasattr(global_timer, "inst_result"):
                #     if not hasattr(global_timer, "inst_result_config"):
                #         global_timer.inst_result_config = []
                #     inst_result = {
                #         "app": app,
                #         "exec": exec,
                #     }
                #     inst_result.update(global_timer.inst_result)
                #     global_timer.inst_result_config.append(inst_result)
                #     del global_timer.inst_result

    finally:
        if not args.dry_run:
            global_timer.display_timings()
            if global_timer.timing_data:
                result_path = os.path.join(os.environ["BITGEN_ROOT"], "raw_results/ac")
                print(process_two_stage(df=global_timer.get_pd_data()))
                global_timer.save_to_file(result_path)
                global_timer.reset_timings()

            # Save inst stats
            if hasattr(global_timer, "inst_result_config"):
                inst_result_config_pd = pd.DataFrame(
                    global_timer.inst_result_config
                )
                print("\n------ Inst Stats ------")
                print(inst_result_config_pd)
                print("-------------------------")
                inst_result_config_path = os.path.join(
                    os.environ["BITGEN_ROOT"], "raw_results/ac_inst_stats"
                )
                global_timer.save_to_file(
                    inst_result_config_path, inst_result_config_pd
                )
            if hasattr(global_timer, "profile_data"):
                profile_result_pd = pd.DataFrame(global_timer.profile_data)
                print("\n------ Profile Stats ------")
                print(profile_result_pd)
                print("-------------------------")
                profile_result_path = os.path.join(
                    os.environ["BITGEN_ROOT"], "raw_results/ac_profile"
                )
                global_timer.save_to_file(profile_result_path, profile_result_pd)
