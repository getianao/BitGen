from .bitparser import BitstreamParser
from .codegen import create_generator
from .bitstream import Bitstream
from .passes import *
from . import config as cfg
from .tool.pcre2pablo import (
    pcre2pablo,
    pcrefile2pablo,
    run_icgrep,
    run_icgrep_from_file,
)
from .tool import global_timer
from .log import MyLogger

import argparse
from colorama import Fore, Style
import os
import re
import logging
import importlib
import sys
import torch
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import shutil

def load_input_stream(input_path):
   with open(input_path, "rb") as f:
        input_stream = f.read()
        return input_stream

def get_icgrep_time(output):
    time = re.search(r"Time:\s*([\d.]+)\s*ms", output)
    if time:
        time = time.group(1)
    else:
        raise Exception("Failed to get icgrep time.")
    return float(time)


def get_icgrep_count(output):
        count = re.search(r"match_count\s*:\s*(\d+)", output)
        if count:
            count = count.group(1)
        else:
            raise Exception(f"Failed to get icgrep count.\noutput: {output}")
        return int(count)

def get_throughput(output):
    throughput = re.search(r"Throughput:\s*([\d.]+)\s*MB/s", output)
    if throughput:
        throughput = throughput.group(1)
    else:
        raise Exception("Failed to get throughput.")
    return throughput


def validation(regex, input_path, result, as_one_group=False):

    def validation_with_icgrep(regex, input_path, regex_from_file=False):
        print(f"Run icgrep")
        if regex_from_file:
            icgrep_output = run_icgrep_from_file(
                regex,
                input_path,
                repeat=cfg.get_config("repeat_input"),
                input_size=cfg.get_config("input_size"),
            )
        else:
            icgrep_output = run_icgrep(
                regex,
                input_path,
                repeat=cfg.get_config("repeat_input"),
                input_size=cfg.get_config("input_size"),
            )
        # print(f"icgrep_output: {icgrep_output.stdout}")
        # print(f"icgrep_count: {icgrep_count}, icgrep_throughput: {icgrep_throughput}")
        icgrep_count = get_icgrep_count(icgrep_output.stdout)
        icgrep_throughput = get_throughput(icgrep_output.stdout)
        return icgrep_count

    multi_input = cfg.get_config("multi_input")
    validation_count = 0
    if isinstance(regex, list):
        # Regex group list
        if as_one_group:
            flattened_regex = [r for regex_group in regex for r in regex_group]
            regex_file_path = "/tmp/regex_group.txt"
            with open(regex_file_path, "w") as f:
                f.write("\n".join(flattened_regex))
            for multi_input_id in range(multi_input):
                validation_count += validation_with_icgrep(
                    regex_file_path, input_path, regex_from_file=True
                )
        else:
            for multi_input_id in range(multi_input):
                for regex_group in regex:
                    regex_file_path = "/tmp/regex_group.txt"
                    with open(regex_file_path, "w") as f:
                        f.write("\n".join(regex_group))
                    validation_count += validation_with_icgrep(
                        regex_file_path, input_path, regex_from_file=True
                    )
    else:
        # Regex string
        assert isinstance(regex, str)
        for multi_input_id in range(multi_input):
            validation_count += validation_with_icgrep(
                regex, input_path, regex_from_file=False
            )

    global_timer.append_data("run_regex", "ref_count", validation_count)
    if validation_count != result.get_count():
        print(
            f"{Fore.RED}Validation Failed: bs_count: {result.get_count()}, icgrep_count: {validation_count}{Style.RESET_ALL}"
        )
        return False
    else:
        print(f"Validation Passed: count {result.get_count()}")
        return True


def run_passes(insts, var_name_map):
    pm = PassManager()
    pm.add_pass(
        print_computation_graph_pass(save_path="computation_graph_0.pdf"),
        enabled=cfg.get_config("pass_print_graph"),
    )
    pm.add_pass(
        cc_advance_pass(),
        enabled=cfg.get_config("pass_cc_advanced"),
    )
    pm.add_pass(
        remove_alias_pass(),
        enabled=True,
    )
    pm.add_pass(
        print_computation_graph_pass(save_path="computation_graph_1.pdf"),
        enabled=cfg.get_config("pass_print_graph"),
    )
    pm.add_pass(
        short_circuit_pass(profile=False),
        enabled=cfg.get_config("pass_short_circuit"),
    )
    # pm.add_pass(
    #     pass_loop_peeling(),
    #     enabled=cfg.get_config("pass_loop_peeling"),
    # )
    # pm.add_pass(
    #     free_intermediate_tensor_pass(),
    #     enabled=cfg.get_config("pass_free_intermediate_tensor"),
    # )
    if cfg.get_config("backend") == "cuda":
        pm.add_pass(
            graph_break_pass(graph_break_type=cfg.get_config("pass_graph_break")),
            enabled=True,
        )
    pm.add_pass(
        inst_stats_pass(),
        enabled=cfg.get_config("pass_inst_stats"),
    )

    # pm.print_passes()
    insts, var_name_map = pm.run(insts, var_name_map)
    return insts


def init_kernel(regex_func_num, kernel_path, input_stream_tensor, basic_stream_tensor):
    def get_attr(func_name, module):
        attr_list = func_name.split(".")
        for attr in attr_list:
            module = getattr(module, attr)
            if module is None:
                raise Exception(f"Attribute {attr} not found in {func_name}")
        return module
    module_path = os.path.dirname(kernel_path)
    module_path = os.path.dirname(module_path)
    sys.path.insert(0, module_path)
    if 'kernels' in sys.modules:
        del sys.modules['kernels']
    import kernels
    importlib.reload(kernels)
    kernel_name = os.path.basename(kernel_path).split(".")[0]
    importlib.reload(get_attr(kernel_name, kernels))
    print(f"Inited kernel: {kernel_name}")
    kernel_init_func_name = f"{kernel_name}.init_kernel"
    kernel_init_func = get_attr(kernel_init_func_name, kernels)

    result, tmp_streams, kernel_handle, kernel, dyn_stats = kernel_init_func(
        regex_func_num, input_stream_tensor, basic_stream_tensor, cfg.get_config_dict()
    )
    return result, tmp_streams, kernel_handle, kernel, dyn_stats


def create_symbolic_link_to_code():
    symlink_name = "generated_code"
    if os.path.islink(symlink_name):
        os.unlink(symlink_name)
    elif os.path.exists(symlink_name):
        raise FileExistsError(
            f"Cannot create symbolic link because '{symlink_name}' is a regular file or directory."
        )
    __cached_module_path = cfg.get_config("__cached_module_path")
    assert __cached_module_path is not None
    os.symlink(__cached_module_path, symlink_name)


def run_kernels(
    kernels,
    input_stream_tensor,
    basic_stream_tensor=None,
    results=None,
    cuda_streams=None,
):
    with torch.no_grad():
        for idx in range(len(kernels)):
            with torch.cuda.stream(cuda_streams[idx]):
                kernels[idx](basic_stream_tensor, results[idx])


def pablo2kernel(
    pablo_regex,
    pablo_path,
    input_path,
    input_stream_tensor,
    basic_stream_tensor=None,
):
    # Parse Pablo to instruction
    parser = BitstreamParser()
    insts = parser.parse_file(pablo_path)
    insts = run_passes(insts, parser.var_name_map)

    # Generate kernel code from instruction
    backend_type = cfg.get_config("backend", default="torch")
    # backend_type = "torch"
    print(f"Backend: {backend_type}")
    Gen = create_generator(backend_type)
    gen = Gen([insts], [parser.var_name_map], regex_list=[[pablo_regex]])
    kernel_path = gen.lower(input_path=input_path)
    gen.close()

    # For debug, create symbolic link to the cache file
    create_symbolic_link_to_code()
    return kernel_path


def pablo_list2kernel(
    pablo_regexes,
    pablo_paths,
    input_path,
    input_stream_tensor,
    basic_stream_tensor=None,
):
    # pablo to IR
    insts_list = []
    var_name_map_list = []
    parallel_pablo2kernel = False
    if not cfg.get_config("use_cached_code"):
        assert isinstance(pablo_paths, list)
        # Parse Pablo to instruction
        def compile_pablo(pablo_id, pablo_path):
            try:
                MyLogger.debug(f"[Compilation] Parse Pablo {pablo_id}: {pablo_path}")
                parser = BitstreamParser()
                insts = parser.parse_file(pablo_path)
                insts = run_passes(insts, parser.var_name_map)
            except Exception as e:
                MyLogger.error(f"Failed to parse Pablo: {pablo_id}: {pablo_path}")
                MyLogger.error(f"Regex: {pablo_regexes[pablo_id]}")
                raise e
            return insts, parser.var_name_map

        if parallel_pablo2kernel:
            raise NotImplementedError("Parallel pablo2kernel is not supported.")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks to the executor
                futures = {
                    executor.submit(compile_pablo, pablo_id, pablo_path): pablo_id
                    for pablo_id, pablo_path in enumerate(pablo_paths)
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Compiling",
                ):
                    pablo_id = futures[future]
                    result = future.result()
                    if result:
                        insts_list.append(result[0])
                        var_name_map_list.append(result[1])
        else:
            for pablo_id, pablo_path in tqdm(
                enumerate(pablo_paths),
                total=len(pablo_paths),
                desc="Compiling",
            ):
                result = compile_pablo(pablo_id, pablo_path)
                insts_list.append(result[0])
                var_name_map_list.append(result[1])

    # Lower IR to kernel code
    backend_type = cfg.get_config("backend", default="cuda")
    print(f"Backend: {backend_type}")
    Gen = create_generator(backend_type)
    gen = Gen(insts_list, var_name_map_list, regex_list=pablo_regexes)
    create_symbolic_link_to_code()
    kernel_path = gen.lower(input_path=input_path)
    # For debug, create symbolic link to the cache file

    return kernel_path


def init_cached_path():
    # Same cached path in one run
    # e.g. .cache_bitgen/bitgen_code_2021_0713_123456
    if not cfg.get_config("__cached_path"):
        if cfg.get_config("use_cached_code"):
            # select last one
            __cached_path = os.listdir(
                os.path.join(cfg.get_config("cache_dir", ".cache_bitgen"))
            )
            __cached_path = sorted(__cached_path)[-1]
        else:
            format_date = datetime.now().strftime("%Y_%m%d_%H%M%S")
            __cached_path = f"bitgen_code_{format_date}"
        __cached_path = os.path.join(
            cfg.get_config("cache_dir", ".cache_bitgen"), __cached_path
        )
        cfg.set_config("__cached_path", __cached_path)


def init_cached_module_path():
    # Different cached module path for different regex in one run
    # e.g. .cache_bitgen/bitgen_code_2021_0713_123456/code, .cache_bitgen/bitgen_code_2021_0713_123456/code_1
    __cached_path = cfg.get_config("__cached_path")
    __cached_module_path = os.path.join(__cached_path, "code")
    if cfg.get_config("use_cached_code"):
        assert os.path.exists(os.path.join(os.getcwd(), __cached_module_path))
    else:
        if os.path.exists(os.path.join(os.getcwd(), __cached_module_path)):
            i = 1
            while os.path.exists(
                os.path.join(os.getcwd(), f"{__cached_module_path}_{i}")
            ):
                i += 1
            __cached_module_path = f"{__cached_module_path}_{i}"
    cfg.set_config("__cached_module_path", __cached_module_path)


def compile_regex(
    regex_str,
    input_path,
    input_stream_tensor,
    basic_stream_tensor=None,
):
    # Code will generate in $cache_dir/$__cached_path/$__cached_module_path/kernels/
    # for example: .cache_bitgen/bitgen_code_2021_0713_123456/code/kernels_1/kernel_0.py
    init_cached_path()
    init_cached_module_path()

    # Compile regex
    print("pcre2pablo")
    pablo_path = pcre2pablo(regex_str, "/tmp/code.pablo")
    print("pablo2kernel")
    kernel_path = pablo2kernel(
        regex_str,
        pablo_path,
        input_path,
        input_stream_tensor,
        basic_stream_tensor,
    )

    # Cache pablo file
    cache_pablo_path = os.path.join(
        os.getcwd(), cfg.get_config("__cached_module_path"), "kernels/pablo"
    )
    os.makedirs(
        cache_pablo_path,
        exist_ok=True,
    )
    shutil.copy(
        pablo_path,
        os.path.join(cache_pablo_path, os.path.basename(pablo_path)),
    )

    print(f"Kernel path: {kernel_path}")
    return kernel_path

def compile_xml(
    regex_str,
    input_path,
    input_stream_tensor,
    basic_stream_tensor=None,
):
    init_cached_path()
    init_cached_module_path()

    # Compile regex
    print("pcre2pablo")
    pablo_path = os.path.join(os.path.dirname(__file__), "bitparser/xml_lexical_parsing.pablo")
    print("pablo2kernel")
    kernel_path = pablo2kernel(
        "xml_parser",
        pablo_path,
        input_path,
        input_stream_tensor,
        basic_stream_tensor,
    )

    # Cache pablo file
    cache_pablo_path = os.path.join(
        os.getcwd(), cfg.get_config("__cached_module_path"), "kernels/pablo"
    )
    os.makedirs(
        cache_pablo_path,
        exist_ok=True,
    )
    shutil.copy(
        pablo_path,
        os.path.join(cache_pablo_path, os.path.basename(pablo_path)),
    )

    print(f"Kernel path: {kernel_path}")
    return kernel_path

def compile_regex_group(regex_group_id, regex_group):
    # MyLogger.debug(f"Compile regex group: {regex_group_id}")
    pablo_path = f"/tmp/code_{regex_group_id}.pablo"
    regex_group_file = f"/tmp/regex_group_{regex_group_id}.txt"

    # Write the regex group to a file
    with open(regex_group_file, "w") as f:
        f.write("\n".join(regex_group))

    # Call pcrefile2pablo for this regex group
    output = pcrefile2pablo(regex_group_file, pablo_path)
    if output.returncode != 0:
        MyLogger.warning(
            f"Failed to compile regex group {regex_group_id}. stderr\n: {output.stderr}"
        )
        return None
    return pablo_path


def compile_regex_group_list(
    regex_group_list,
    input_path,
    input_stream_tensor,
    basic_stream_tensor=None,
):
    # Code will generate in $cache_dir/$__cached_path/$__cached_module_path/kernels/
    # for example: .cache_bitgen/bitgen_code_2021_0713_123456/code/kernels_1/kernel_0.py
    init_cached_path()
    init_cached_module_path()

    #  Compile regex group list
    pablo_paths = [None] * len(regex_group_list)
    pablo_regexes = [None] * len(regex_group_list)
    print("pcrefile2pablo")

    parallel = cfg.get_config("parallel_compile_parabix", True)
    if not cfg.get_config("use_cached_code"):
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks to the executor
                futures = {
                    executor.submit(
                        compile_regex_group, regex_group_id, regex_group
                    ): regex_group_id
                    for regex_group_id, regex_group in enumerate(regex_group_list)
                }
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Compiling",
                ):
                    regex_group_id = futures[future]
                    result = future.result()
                    if result:
                        pablo_paths[regex_group_id] = result
                        pablo_regexes[regex_group_id] = regex_group_list[regex_group_id]

        else:
            for regex_group_id, regex_group in tqdm(
                enumerate(regex_group_list),
                total=len(regex_group_list),
                desc="Compiling",
            ):
                pablo_path = compile_regex_group(regex_group_id, regex_group)
                if pablo_path:
                    pablo_paths[regex_group_id] = pablo_path
                    pablo_regexes[regex_group_id] = regex_group

        MyLogger.info(f"[pcre2pablo] Compiled {len(pablo_paths)} regex groups.")
        if len(pablo_paths) < len(regex_group_list):
            MyLogger.warning(
                f"[pcre2pablo] Summary: Failed to compile {len(regex_group_list) - len(pablo_paths)} regex groups."
            )

    print("pablo2kernel")
    kernel_path = pablo_list2kernel(
        pablo_regexes,
        pablo_paths,
        input_path,
        input_stream_tensor,
        basic_stream_tensor,
    )
    if not cfg.get_config("use_cached_code"):
        # Cache pablo file
        cache_pablo_path = os.path.join(
            os.getcwd(), cfg.get_config("__cached_module_path"), "kernels/pablo"
        )
        os.makedirs(
            cache_pablo_path,
            exist_ok=True,
        )
        for pablo_path in pablo_paths:
            shutil.copy(
                pablo_path,
                os.path.join(cache_pablo_path, os.path.basename(pablo_path)),
            )

    print(f"Kernel path: {kernel_path}")
    return kernel_path


def run_regex_kernel(
    kernel_path,
    regex,
    input_path,
    input_stream_tensor,
    basic_stream_tensor=None,
):
    # Get kernel
    if isinstance(regex, list):
        regex_func_num = len(regex)
    elif isinstance(regex, str):
        regex_func_num = 1
    else:
        raise Exception("Invalid regex type.")
    result, tmp_streams, kernel_handle, kernel, dyn_stats = init_kernel(
        regex_func_num, kernel_path, input_stream_tensor, basic_stream_tensor
    )

    def reset_tensor(result, tmp_streams):
        if result is not None:
            result.zero_()
        if tmp_streams is not None:
            tmp_streams.zero_()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # print(kernel.output())

    # x = torch._dynamo.explain(kernel)(basic_stream_tensor)
    # print("break_reasons", x.break_reasons)
    # print("graph_break_count", x.graph_break_count)
    # print("graph_count", x.graph_count)
    # print("op_count", x.op_count)
    # # print("ops_per_graph", x.ops_per_graph)
    # # print("out_guards", x.out_guards)
    # print("compile_times", x.compile_times)

    # from .template.template_torch import print_ts_as_bs
    # print_ts_as_bs(input_stream_tensor[0][0], msg="input:")
    # for i in range(8):
    #     print_ts_as_bs(basic_stream_tensor[0][0][i], msg=f"basis[{i}]:")

    # Warm up
    warmup_iters = cfg.get_config("warmup_iters")
    exec_iters = cfg.get_config("exec_iters")
    print(f"Warm up: {warmup_iters} iters")
    print(f"Run: {exec_iters} iters")
    with torch.no_grad():
        for exe_iter in range(warmup_iters + exec_iters):
            reset_tensor(result, tmp_streams)
            if exe_iter < warmup_iters:
                r = kernel(basic_stream_tensor)
            else:
                with global_timer.time("run_regex"):
                    r = kernel(basic_stream_tensor)

    input_size = (
        basic_stream_tensor.numel() * basic_stream_tensor.element_size() / 1024 / 1024
    )  # MB
    global_timer.append_data("run_regex", "input_size", input_size)
    avg_duration = global_timer.get_value("run_regex", "avg_duration")
    throughput = input_size / avg_duration * 1024  # MB/s
    global_timer.append_data("run_regex", "throughput", throughput)
    # Validation
    if cfg.get_config("validation"):
        print(f"Validation")
        # Result
        reset_tensor(result, tmp_streams)
        if cfg.get_config("backend") == "cuda":
            kernel(basic_stream_tensor)
        else:
            result = kernel(basic_stream_tensor)
        bs_result = Bitstream(result.cpu().numpy())
        # bs_result.save(
        #     os.path.join(os.getcwd(), cfg.get_config("__cached_module_path"), "kernels/result.txt")
        # )
        result_count = bs_result.get_count()

        print("{:<32}{}".format("Result_bs: ", bs_result))
        print(f"Count bs: {result_count}")
        for nz_id, nz in enumerate(bs_result.get_count_array()):
            print(f"Count array {nz_id}: {nz}")
        # print(f"Nonzero positions: {result.get_nonzero_positions()}, length: {len(result.data)}")
        global_timer.append_data("run_regex", "count", result_count)
        if regex == "xml_parser":
            pass
        else:
            pass_check = validation(regex, input_path, bs_result, as_one_group=True)
            global_timer.append_data("run_regex", "check", pass_check)
            # if not pass_check:
            #     raise Exception("Validation failed.")
    if cfg.get_config("pass_inst_stats_dyn"):
        iter_num = dyn_stats[..., :1]
        dyn_offset_num = dyn_stats[..., 1:]
        print("iter_num:", iter_num.shape, iter_num.sum().item())
        print("dyn_offset_num:", dyn_offset_num.shape, dyn_offset_num.sum().item())
        global_timer.inst_result["iter_num_total"] = (
            global_timer.inst_result.get("iter_num_total", 0) + iter_num.sum().item()
        )
        global_timer.inst_result["dyn_offset_num_max"] = (
            global_timer.inst_result.get("dyn_offset_num_max", 0)
            + dyn_offset_num.to(torch.float32).max().item()
        )
        global_timer.inst_result["dyn_offset_num_std"] = (
            dyn_offset_num.to(torch.float32).std().item()
        )
        global_timer.inst_result["dyn_offset_num_total"] = (
            global_timer.inst_result.get("dyn_offset_num_total", 0)
            + dyn_stats.sum().item()
        )

        print(dyn_stats.shape)
        print(dyn_stats)
        print(global_timer.inst_result)
    torch.cuda.synchronize()
    result = None
    tmp_streams = None
    torch.cuda.empty_cache()
    if kernel_handle is not None:
        kernel_handle.free()


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Automata compiler")
    arg_parser.add_argument(
        "-r", default=False, action="store_true", help="Use cached code."
    )
    args = arg_parser.parse_args()

    # input_path = "/home/tge/workspace/bitstream-project/datasets/ANMLZoo/Snort/inputs/snort_1MB.input"
    input_path = "/home/tge/workspace/automata-compiler/datasets/small/input/apple.txt"
    input_stream = load_input_stream(input_path)

    # regexes_file = "/home/tge/workspace/bitstream-project/datasets/ANMLZoo/Snort/regex/snort.1chip.filter.regex"
    # run_regex_file(regexes_file, input_path,,input_stream, args.r)

    regex = "apple"
    count = run_regex(regex, input_path, input_stream)
