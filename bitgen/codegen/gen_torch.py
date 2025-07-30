import os
from itertools import chain

from .gen import Generator
from ..inst import BsInstType
from ..inst import BsAdvance
from .. import config as cfg


inst_module_name_map = {
    BsInstType.AND: "BSAnd",
    BsInstType.OR: "BSOr",
    BsInstType.NOT: "BSNot",
    BsInstType.XOR: "BSXor",
    BsInstType.ADD: "BSAdd",
    BsInstType.MATCHSTAR: "BSMatchStar",
    BsInstType.SCANTHRU: "BSScanThru",
    # BsInstType.ASSIGN: "BSAssign",
    BsInstType.SEL: "BSSel",
    BsInstType.IF: "BSIf",
    BsInstType.WHILE: "BSWhile",
    BsInstType.ADVANCE: "BSAdvance",
    BsInstType.TERNARY: "BSTernary",
    BsInstType.MATCH: "BSMatch",
    # BsInstType.CALL: "BSCall",
    BsInstType.BLOCK: "BSBlock",
    BsInstType.STATE: "BSState",
}

inst_name_map = {
    BsInstType.AND: "and",
    BsInstType.OR: "or",
    BsInstType.NOT: "not",
    BsInstType.XOR: "xor",
    BsInstType.ADD: "add",
    BsInstType.MATCHSTAR: "matchstar",
    BsInstType.SCANTHRU: "scanthru",
    BsInstType.ASSIGN: "assign",
    BsInstType.SEL: "sel",
    BsInstType.IF: "if",
    BsInstType.WHILE: "while",
    BsInstType.ADVANCE: "advance",
    BsInstType.TERNARY: "ternary",
    BsInstType.MATCH: "match",
    BsInstType.CALL: "call",
    BsInstType.BLOCK: "block",
    BsInstType.STATE: "state",
}


class TorchGenerator(Generator):

    def __init__(self, insts_list, var_name_map_list, regex_list=None):
        super().__init__(insts_list, var_name_map_list, regex_list)
        if len(insts_list) > 1:
            raise NotImplementedError("TorchGenerator only supports one kernel")
        self.torch_file_handler = None
        self.module_map = {}  # inst_id -> module_name
        self.module_type_num_map = {}  # module_type -> number
        self.advance_count = {}  # var_id -> count

    def close(self):
        if self.torch_file_handler is not None:
            self.torch_file_handler.close()

    def __del__(self):
        self.close()

    def add_module(self, inst_id, inst, indent=""):
        if inst.type == BsInstType.IF:
            self.lower_module(list(chain(inst.body_1.body)), indent)
            if inst.body_2 is not None:
                self.lower_module(list(chain(inst.body_2.body)), indent)
            return ""

        if inst.type == BsInstType.WHILE:
            self.lower_module(list(chain(inst.body.body)), indent)
            return ""

        if inst_id in self.module_map:
            assert "module_name should not be generated twice"
        if inst.type in self.module_type_num_map:
            self.module_type_num_map[inst.type] += 1
        else:
            self.module_type_num_map[inst.type] = 0
        module_number = self.module_type_num_map[inst.type]
        module_name = inst_name_map[inst.type] + "_" + str(module_number)
        self.module_map[inst_id] = module_name
        return module_name

    def lower_module(self, insts, indent=""):
        if insts is None or len(insts) == 0:
            return
        code_lines = []

        for idx, inst in enumerate(insts):
            if inst.operation != "":
                continue
            if inst.type not in inst_module_name_map:
                continue
            module_name = self.add_module(idx, inst, indent)
            if module_name == "":
                continue
            inst.operation = "self." + module_name
            module_define = (
                f"{indent}{inst.operation} = " + inst_module_name_map[inst.type] + "()"
            )

            code_lines.append(module_define)
        code_lines.append("")
        self.torch_file_handler.write("\n".join(code_lines))

    def lower_insts(self, insts, var_name_map, indent=""):
        if insts is None or len(insts) == 0:
            return
        code_lines = []
        for inst in insts:
            code_lines.append(inst.lower_to_torch(var_name_map, indent))
        code_lines.append("")
        self.torch_file_handler.write("\n".join(code_lines))

    def lower_regex_function(self):
        for insts_id in range(len(self.insts_list)):
            insts = self.insts_list[insts_id]
            var_name_map = self.var_name_map_list[insts_id]
            indent = "      "
            string_define = [
                "",
                "class KernelGenerated(nn.Module):",
                f"{indent}def __init__(self, n_char):",
                f"{indent}{indent}super(KernelGenerated, self).__init__()",
                f"{indent}{indent}self.n_char = n_char",
                "",
            ]
            self.torch_file_handler.write("\n".join(string_define))
            self.lower_module(insts, indent * 2)

            string_define_2 = [
                "",
                # f"{indent}@mytimer(warmup_runs=3, runs=1)",
                f"{indent}# @mem_profile",
                f"{indent}def forward(self, basic_stream, result=None):",
                "",
            ]
            self.torch_file_handler.write("\n".join(string_define_2))
            self.lower_insts(insts, var_name_map, indent * 2)

            string_kernel_end = [
                f"{indent}{indent}return bs_result\n",
                "",
            ]
            self.torch_file_handler.write("\n".join(string_kernel_end))

    def lower_kernel(self):
        with open(
            "bitgen/template/template_torch.py", "r", encoding="utf-8"
        ) as f:
            template = f.read()
            self.torch_file_handler.write(template)

        self.lower_regex_function()

    def lower(self, input_path=None):
        # Initialize kernel file
        kerne_file_name = "kernel_0.py"
        kernel_file_path = self.get_kernel_path(kerne_file_name)
        if cfg.get_config("use_cached_code"):
            print(f"Use cached code: {kernel_file_path}")
            return kernel_file_path
        else:
            self.torch_file_handler = open(kernel_file_path, "w", encoding="utf-8")
            print(f"Generate code: {kernel_file_path}")

        self.lower_kernel()
        self.lower_main(input_path)
        self.gen_kernel_module_file([kerne_file_name])

        return kernel_file_path

    def lower_main(self, input_path):
        indent = "      "
        main_codes = [
            "if __name__ == '__main__':",
            f"{indent}try:",
            f"{indent}{indent}input_path = '{input_path}'",
            f"{indent}{indent}input_stream = load_input_stream(input_path)",
            f"{indent}{indent}input_stream_tensor = torch.ByteTensor(list(input_stream))",
            f"{indent}{indent}num_repeats = 32",
            f"{indent}{indent}input_stream_tensor = input_stream_tensor.repeat(num_repeats)",
            f"{indent}{indent}input_stream_tensor = torch.ByteTensor(list(input_stream))",
            f"{indent}{indent}exec_kernel(input_stream_tensor=input_stream_tensor, basic_stream_tensor=None)",
            f"{indent}finally:",
            # f"{indent}{indent}result_path = os.path.join(os.environ["BITGEN_ROOT"], "raw_results/ac")
            # f"{indent}{indent}global_timer.save_to_file(result_path)",
            f"{indent}{indent}global_timer.display_timings()",
            f"{indent}{indent}global_timer.reset_timings()",
        ]
        self.torch_file_handler.write("\n".join(main_codes))
