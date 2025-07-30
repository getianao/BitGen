import os
import math
from itertools import chain
import shutil

from .gen import Generator
from ..inst import *
from ..inst.inst import get_max_advance_offset
from .. import config as cfg
from ..log import MyLogger
from ..passes.pass_utils import break_insts
from ..tool import global_timer


class CudaGenerator(Generator):

    def __init__(self, insts_list, var_name_map_list, regex_list=None):
        super().__init__(insts_list, var_name_map_list, regex_list)
        self.kernel_wrapper_file_handler = None
        self.kernel_file_handler = None
        self.tmp_streams_num = 0
        self.parallel_compile = cfg.get_config("parallel_compile_cuda")

    def close(self):
        if self.kernel_wrapper_file_handler is not None:
            self.kernel_wrapper_file_handler.close()
            self.kernel_wrapper_file_handler = None
        if self.kernel_file_handler is not None:
            self.kernel_file_handler.close()
            self.kernel_file_handler = None

    def __del__(self):
        self.close()

    def get_var_by_name(self, var_name, var_name_map):
        for var in var_name_map:
            if var_name_map[var] == var_name:
                return var
        return None

    def get_max_advance_num(self, insts):
        max_advance_offset = 0
        # if cfg.get_config("pass_cc_advanced"):
        #     for inst in insts:
        #         if inst.type == BsInstType.ADVANCE:
        #             max_advance_offset = max(max_advance_offset, inst.operand2)
        # else:
        for inst in insts:
            if inst.type == BsInstType.ADVANCE:
                # if "Right" in inst.operation:
                #     if inst.dymatic_offset:
                #         max_advance_offset += 1
                #     else:
                #         max_advance_offset = max(max_advance_offset, inst.operand2)
                # elif "Left" in inst.operation:
                #     max_advance_offset += 1
                max_advance_offset += 1
        return max_advance_offset

    def lower_kernel_wrapper(self):
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "backend",
            "cuda",
            "template",
            (
                "template_dyn_stat.py"
                if cfg.get_config("pass_inst_stats_dyn")
                else "template.py"
            ),
        )
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
            self.kernel_wrapper_file_handler.write(template + "\n\n")
        if self.tmp_streams_num == 0:
            self.tmp_streams_num = 1
        self.kernel_wrapper_file_handler.write(
            f"tmp_streams_num = {self.tmp_streams_num}\n"
        )
        if cfg.get_config("pass_inst_stats"):
            global_timer.inst_result["tmem"] = (
                global_timer.inst_result.get("tmem", 0) + self.tmp_streams_num
            )

        self.kernel_wrapper_file_handler.write(
            f"grid_size = {cfg.get_config('grid_size')}  # regex_num, input_num, 1\n"
        )
        self.kernel_wrapper_file_handler.write(
            f"block_size = {cfg.get_config('block_size')}\n"
        )

    def low_insts(self, insts, var_name_map, var_define_map, indent, file_handler):
        code_lines = []
        for inst_id, inst in enumerate(insts):
            inst_string = inst.lower_to_cuda(var_name_map, indent, var_define_map)
            # First 8 instructions are for bistream assignment
            # if inst_id < 8:
            #     comment_pos = inst_string.find("//")
            #     if comment_pos == -1:
            #         comment_pos = len(inst_string)
            #     inst_string = (
            #         inst_string[:comment_pos].rstrip()[:-2]
            #         + " * n_unit_basic + i];"
            #         + inst_string[comment_pos:]
            #     )
            if inst_string is not None:
                code_lines.append(inst_string)
        self.add_lines(file_handler, code_lines)

    def lower_subgraph_scanthru(
        self, insts, var_name_map, var_define_map, output_var, indent, file_handler
    ):
        inst_scanthru = None
        for insts_id, inst in enumerate(insts):
            if inst.type == BsInstType.SCANTHRU:
                inst_scanthru = inst
                break
        assert inst_scanthru is not None
        op1_name = var_name_map[inst_scanthru.operand1]
        op2_name = var_name_map[inst_scanthru.operand2]
        ret_name = var_name_map[inst_scanthru.ret]
        op1_stream_id = self.get_var_by_name(op1_name + "_stream", var_name_map)
        op2_stream_id = self.get_var_by_name(op2_name + "_stream", var_name_map)
        ret_stream_id = self.get_var_by_name(ret_name + "_stream", var_name_map)
        inst_scanthru_stream = BsScanThruStream(
            [op1_stream_id, op2_stream_id], ret_stream_id
        )
        self.low_insts(
            [inst_scanthru_stream], var_name_map, var_define_map, indent, file_handler
        )

    def lower_subgraph(
        self, insts, var_name_map, var_define_map, output_var, indent, file_handler
    ):
        graph_break_type = cfg.get_config("pass_graph_break")
        indent2 = indent + "    "
        max_advance_num = self.get_max_advance_num(insts)
        max_advance_offset = math.ceil(max_advance_num / 32)
        block_size = cfg.get_config("block_size")[0]

        if cfg.get_config("pass_inst_stats"):
            global_timer.inst_result["static_advance_num_total"] = (
                global_timer.inst_result.get("static_advance_num_total", 0) + max_advance_num
            )
            global_timer.inst_result["loop"] = (
                global_timer.inst_result.get("loop", 0) + 1
            )

        if graph_break_type == -1:
            # Fuse all.
            if cfg.get_config("pass_inst_stats_dyn"):
                input_loop_start = [
                    f"{indent}__shared__ uint32_t dynamic_adv_offset;",
                    f"{indent}__shared__ uint32_t zero_flag;",
                    f"{indent}uint32_t iter_number = 0;",
                    f"{indent}uint32_t loop_counter = 0;",
                    f"{indent}for (uint32_t idx = threadIdx.x; idx < n_unit_basic + threadIdx.x; idx = idx + blockDim.x - (dynamic_adv_offset + {max_advance_num} + 32) / 32) {{",
                    f"{indent2}__syncthreads();",
                    f"{indent2}if (threadIdx.x == 0) {{ dynamic_adv_offset = 0; }}",
                ]

                input_loop_end = [
                    f"{indent2}__syncthreads();", # Make sure dynamic_adv_offset is updated. 
                    f"{indent2}// assert (dynamic_adv_offset <= {block_size}*32);",
                    f'{indent2}// if (threadIdx.x == 0 && dynamic_adv_offset >= {block_size}*32) {{printf("dynamic_adv_offset error: %d\\n", dynamic_adv_offset);}}',
                    f'{indent2}// if (threadIdx.x == 0) {{printf("dynamic_adv_offset: %d\\n", dynamic_adv_offset);}}',
                    f"{indent2}if (threadIdx.x == 0) {{ dyn_stats[iter_number + 1] = dynamic_adv_offset; iter_number++;}}",
                    f"{indent}}}",
                    f"{indent}if (threadIdx.x == 0) {{ dyn_stats[0] = iter_number; }}",
                ]

            else:
                input_loop_start = [
                    f"{indent}__shared__ uint32_t dynamic_adv_offset;",
                    f"{indent}__shared__ uint32_t zero_flag;",
                    f"{indent}uint32_t loop_counter = 0;",
                    f"{indent}for (uint32_t idx = threadIdx.x; idx < n_unit_basic + threadIdx.x; idx = idx + blockDim.x - (dynamic_adv_offset + {max_advance_num} + 32) / 32) {{",
                    f"{indent2}__syncthreads();",
                    f"{indent2}if (threadIdx.x == 0) {{ dynamic_adv_offset = 0; }}",
                ]

                input_loop_end = [
                    f"{indent2}__syncthreads();", # Make sure dynamic_adv_offset is updated. 
                    f"{indent2}// assert (dynamic_adv_offset <= {block_size}*32);",
                    f'{indent2}// if (threadIdx.x == 0 && dynamic_adv_offset >= {block_size}*32) {{printf("dynamic_adv_offset error: %d\\n", dynamic_adv_offset);}}',
                    f'{indent2}// if (threadIdx.x == 0) {{printf("dynamic_adv_offset: %d\\n", dynamic_adv_offset);}}',
                    f"{indent}}}",
                ]
        else:
            input_loop_start = [
                f"{indent}for (uint32_t i = 0; i < ceil(1.0 * n_unit_basic / (blockDim.x - {max_advance_offset})); i += 1) {{",
                f"{indent2}int idx = i * (blockDim.x - {max_advance_offset}) + threadIdx.x;",
            ]
            input_loop_end = [f"{indent2}__syncthreads();", f"{indent}}}"]

        insts_type = [inst.type for inst in insts]
        if graph_break_type != -1 and BsInstType.WHILE in insts_type:
            self.low_insts(insts, var_name_map, var_define_map, indent, file_handler)
        elif graph_break_type != -1 and BsInstType.SCANTHRU in insts_type:
            self.lower_subgraph_scanthru(
                insts, var_name_map, var_define_map, output_var, indent, file_handler
            )
        else:
            self.add_lines(file_handler, input_loop_start)
            self.low_insts(insts, var_name_map, var_define_map, indent2, file_handler)
            self.add_lines(file_handler, input_loop_end)
        self.add_lines(file_handler, f"{indent}__syncthreads();")

    # tmp_stream_map : offset, [var_stream_id]
    def reuse_tmp_stream(self, insts, inst_id, var_name_map, tmp_stream_map):
        if inst_id == 0:
            return None
        check_reuse_stream = insts[inst_id].ret
        check_reuse_stream_start = -1
        stream_name = var_name_map[check_reuse_stream]
        if stream_name.startswith("adv"):
            return None

        # def check_stream_name(name, prefix, suffix):
        #     if name.startswith(prefix) and name.endswith(suffix):
        #         return True
        #     return False

        # if check_stream_name(stream_name, "test", "stream_next"):
        #     for tmp_stream_map_offset in tmp_stream_map.keys():
        #         for var_stream_id in tmp_stream_map[tmp_stream_map_offset]:
        #             if check_stream_name(
        #                 var_name_map[var_stream_id], "test", "stream_next"
        #             ):
        #                 return tmp_stream_map_offset
        # if check_stream_name(stream_name, "accum", "stream_next"):
        #     for tmp_stream_map_offset in tmp_stream_map.keys():
        #         for var_stream_id in tmp_stream_map[tmp_stream_map_offset]:
        #             if check_stream_name(
        #                 var_name_map[var_stream_id], "accum", "stream_next"
        #             ):
        #                 return tmp_stream_map_offset
        # return None

        # Get the first definition of check_reuse_stream
        for iii, inst_iii in enumerate(insts):
            if inst_iii.type == BsInstType.IF:
                for sub_inst in inst_iii.body_1.body:
                    if (
                        sub_inst.type == BsInstType.STREAMSTORE
                        and sub_inst.ret == check_reuse_stream
                    ):
                        check_reuse_stream_start = iii
                        break
            # elif inst_iii.type == BsInstType.WHILE:
            #     for sub_inst in inst_iii.body.body:
            #         if (
            #             sub_inst.type == BsInstType.STREAMSTORE
            #             and sub_inst.ret == check_reuse_stream
            #         ):
            #             check_reuse_stream_start = iii
            #             break

        if check_reuse_stream_start == -1:
            return None
        for tmp_offset in reversed(tmp_stream_map.keys()):
            for var_stream_id in tmp_stream_map[tmp_offset]:
                # After definition of check_reuse_stream, var_stream_id cannot be used
                for iii in range(check_reuse_stream_start, len(insts)):
                    inst_iii = insts[iii]
                    if inst_iii.type == BsInstType.WHILE:
                        continue
                    for operand_idx in range(inst_iii.n_operand):
                        if inst_iii.operands[operand_idx] == var_stream_id:
                            return None
            return tmp_offset
        return None

    # Return the number of stream define insts
    def lower_graph(self, insts, var_name_map, var_define_map, indent, file_handler):
        if insts is None or len(insts) == 0:
            return

        # Lower stream define insts before sub-graph
        stream_define_insts = []
        # code_define_carry_stream = f"{indent}uint32_t* carry_stream = tmp_streams + n_unit_basic * {self.tmp_streams_num};"
        # self.tmp_streams_num += 1
        # code_define_add_stream = f"{indent}uint32_t* add_stream = tmp_streams + n_unit_basic * {self.tmp_streams_num};"
        # self.tmp_streams_num += 1
        # self.add_lines(file_handler, code_define_carry_stream)
        # self.add_lines(file_handler, code_define_add_stream)
        tmp_stream_map = {} # tmp_stream_offset, [var_stream_id]
        for inst_id, inst in enumerate(insts):    
            if inst.type == BsInstType.STREAMDEFINE:
                tmp_offset = None
                # tmp_offset = self.reuse_tmp_stream(insts, inst_id, var_name_map, tmp_stream_map)
                # print(
                #     f"reuse_tmp_stream: {var_name_map[insts[inst_id].ret]}, {tmp_offset}, {self.tmp_streams_num}"
                # )
                if tmp_offset is None:
                    tmp_offset = self.tmp_streams_num
                    self.tmp_streams_num += 1
                inst.stream_index = f"n_unit_basic * {tmp_offset}"
                if tmp_offset in tmp_stream_map:
                    tmp_stream_map[tmp_offset].append(inst.ret)
                else:
                    tmp_stream_map[tmp_offset] = [inst.ret]
                stream_define_insts.append(inst)
            elif inst.type == BsInstType.WHILE:
                for sub_inst_id, sub_inst in enumerate(inst.body.body):
                    if sub_inst.type == BsInstType.STREAMDEFINE:
                        insts[inst_id].body.body[
                            sub_inst_id
                        ].stream_index = f"n_unit_basic * {self.tmp_streams_num}"
                        self.tmp_streams_num += 1
        self.low_insts(
            stream_define_insts, var_name_map, var_define_map, indent, file_handler
        )

        # Group insts by graph break
        insts = insts[len(stream_define_insts):]
        insts_break_group = break_insts(insts)
        if cfg.get_config("pass_graph_break") == -1:
            assert len(insts_break_group) == 1

        # Generate code for each group
        for insts_break in insts_break_group:
            output_var = None
            var_define_map_subgraph = var_define_map.copy()
            self.lower_subgraph(
                insts_break,
                var_name_map,
                var_define_map_subgraph,
                output_var,
                indent,
                file_handler,
            )

    def lower_regex_function(self):
        for insts_id in range(len(self.insts_list)):
            if self.parallel_compile:
                # print(f"Lower regex_{insts_id}")
                kernel_file_path = self.get_kernel_path(f"kernel_regex_{insts_id}.cu")
                kernel_regex_file_handler = open(
                    kernel_file_path, "w", encoding="utf-8"
                )
                block_size = cfg.get_config("block_size")[0]
                kernel_regex_file_handler.write(f"#define BLOCK_SIZE {block_size}\n")
                template_path = os.path.join(
                    os.path.dirname(__file__),
                    "../backend/cuda/template/template.cu",
                )
                with open(template_path, "r", encoding="utf-8") as f:
                    template = f.read()
                    kernel_regex_file_handler.write(template + "\n\n")
            else:
                kernel_regex_file_handler = self.kernel_file_handler

            insts = self.insts_list[insts_id]
            regex_str = (
                "["
                + ", ".join(f'"{regex}"' for regex in self.regex_list[insts_id])
                + "]"
            )
            var_name_map = self.var_name_map_list[insts_id]
            var_define_map = {}
            var_result_stream = self.get_var_by_name("bs_result_stream", var_name_map)
            # print("var_result_stream", var_result_stream)
            if var_result_stream is not None:
                var_define_map[var_result_stream] = 1

            if  cfg.get_config("pass_inst_stats_dyn"):
                string_kernel_start = [
                    f"// Regex: {(regex_str)}",
                    f'extern "C" __noinline__ __device__ void regex_{insts_id}(uint32_t* basic_stream, uint32_t n_unit_basic, uint32_t n_char, uint32_t* bs_result_stream, uint32_t* advance_memory, uint32_t* tmp_streams, uint32_t* dyn_stats) {{',
                ]
            else:
                string_kernel_start = [
                    f"// Regex: {(regex_str)}",
                    f'extern "C" __noinline__ __device__ void regex_{insts_id}(uint32_t* basic_stream, uint32_t n_unit_basic, uint32_t n_char, uint32_t* bs_result_stream, uint32_t* advance_memory, uint32_t* tmp_streams) {{',
                ]

            self.add_lines(kernel_regex_file_handler, string_kernel_start)
            self.lower_graph(
                insts,
                var_name_map,
                var_define_map,
                indent="    ",
                file_handler=kernel_regex_file_handler,
            )
            string_kernel_end = [
                "}",
                "",
            ]
            self.add_lines(kernel_regex_file_handler, string_kernel_end)
            if self.parallel_compile:
                kernel_regex_file_handler.close()

    def lower_kernel(self):
        # Write pre-defined template.
        if not self.parallel_compile:
            block_size = cfg.get_config("block_size")[0]
            self.kernel_file_handler.write(f"#define BLOCK_SIZE {block_size}\n")
            template_path = os.path.join(
                os.path.dirname(__file__),
                "../backend/cuda/template/template.cu",
            )
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
                self.kernel_file_handler.write(template + "\n\n")
        # Generate regex as device function.
        self.lower_regex_function()

        # In cuda backend, one block process 1 regex and 1 input stream
        regex_num = len(self.insts_list)
        input_num = cfg.get_config("multi_input", 1)
        cfg.set_config("grid_size", (regex_num, input_num, 1))
        if cfg.get_config("pass_inst_stats"):
            global_timer.inst_result["regex_group_num"] = (
                global_timer.inst_result.get("regex_group_num", 0) + regex_num
            )

        max_advance_offset_all = cfg.get_config("pass_cc_advanced_max")
        if max_advance_offset_all < 1:
            raise NotImplementedError("max_advance_offset_all < 1")

        if self.parallel_compile:
            kernel_link_code = [
                f"typedef unsigned int uint32_t;",
            ]
            for regex_id in range(regex_num):
                if cfg.get_config("pass_inst_stats_dyn"):
                    kernel_link_code.append(
                        f'extern "C" __device__ void regex_{regex_id}(uint32_t* basic_stream, uint32_t n_unit_basic, uint32_t n_char, uint32_t* bs_result_stream, uint32_t* advance_memory, uint32_t* tmp_streams, uint32_t* dyn_stats);'
                    )
                else:
                    kernel_link_code.append(
                        f'extern "C" __device__ void regex_{regex_id}(uint32_t* basic_stream, uint32_t n_unit_basic, uint32_t n_char, uint32_t* bs_result_stream, uint32_t* advance_memory, uint32_t* tmp_streams);'
                    )

            self.add_lines(self.kernel_file_handler, kernel_link_code)

        block_size = cfg.get_config("block_size")[0]

        string_kernel_start = [
            """extern "C" __global__ void KernelGenerated(uint32_t* input_stream, uint32_t n_unit_basic, uint32_t n_char, uint32_t* bs_result_stream, uint32_t* tmp_streams, uint32_t* dyn_stats) {""" if cfg.get_config("pass_inst_stats_dyn") else """extern "C" __global__ void KernelGenerated(uint32_t* input_stream, uint32_t n_unit_basic, uint32_t n_char, uint32_t* bs_result_stream, uint32_t* tmp_streams) {""",
            f"    __shared__ uint32_t advance_memory[{block_size} * {max_advance_offset_all + 1}];", # 0 preserve for sync advance.
            # f"    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;",
            # f"    uint32_t thread_id_in_block = threadIdx.x;",
            f"    input_stream = input_stream + 8 * n_unit_basic * blockIdx.y;",
            f"    tmp_streams = tmp_streams + blockIdx.y * {self.tmp_streams_num} * n_unit_basic;",
            f"    bs_result_stream = bs_result_stream + blockIdx.y * {regex_num} * n_unit_basic + blockIdx.x * n_unit_basic;",
            "",
        ]
        if cfg.get_config("pass_inst_stats_dyn"):
            string_kernel_start.append(
                f"    dyn_stats = dyn_stats + blockIdx.y * {regex_num} * n_unit_basic + blockIdx.x * n_unit_basic;"
            )
        self.add_lines(self.kernel_file_handler, string_kernel_start)

        for regex_id in range(regex_num):
            string_if = "if" if regex_id == 0 else "else if"
            string_invoke_function = [
                f"{string_if} (blockIdx.x == {regex_id}) {{    // regex_{regex_id}",
                f"    regex_{regex_id}(input_stream, n_unit_basic, n_char, bs_result_stream, advance_memory, tmp_streams, dyn_stats);" if cfg.get_config("pass_inst_stats_dyn") else f"    regex_{regex_id}(input_stream, n_unit_basic, n_char, bs_result_stream, advance_memory, tmp_streams);",
                # "if(threadIdx.x == 0) { printf(\"regex_id: %d end\\n\", blockIdx.x); }",
                f"    return;",
                "}",
            ]
            self.add_lines(
                self.kernel_file_handler,
                string_invoke_function,
                indent="    ",
            )

        string_kernel_end = "}\n"
        self.add_lines(self.kernel_file_handler, string_kernel_end)

    def lower(self, input_path=None):
        # Initialize kernel file
        kerne_wrapper_file_name = "kernel_0.py"
        kerne_wrapper_file_path = self.get_kernel_path(kerne_wrapper_file_name)
        kernel_file_name = "kernel_0.cu"
        kernel_file_path = self.get_kernel_path(kernel_file_name)
        if cfg.get_config("use_cached_code"):
            print(f"Use cached code: {kerne_wrapper_file_path}")
            print(f"Use cached code: {kernel_file_path}")
            return kerne_wrapper_file_path
        else:
            self.kernel_wrapper_file_handler = open(
                kerne_wrapper_file_path, "w", encoding="utf-8"
            )
            self.kernel_file_handler = open(kernel_file_path, "w", encoding="utf-8")
            print(f"Generate code: {kerne_wrapper_file_path}")
            print(f"Generate code: {kernel_file_path}")

        self.lower_kernel()
        # Must be called after lower_kernel because of tmp_streams_num
        self.lower_kernel_wrapper()
        self.gen_kernel_module_file([kerne_wrapper_file_name])
        return kerne_wrapper_file_path
