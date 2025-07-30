from itertools import chain

from .gen import Generator

class PythonGenerator(Generator):

    def lower_insts(self, insts, indent=""):
        if insts is None or len(insts) == 0:
            return
        code_lines = []
        for inst in insts:
            code_lines.append(inst.lower_to_python(self.var_name_map, indent))
        code_lines.append("")
        self.file_handler.write("\n".join(code_lines))

    def lower(self):
        with open("bitgen/template/template.py", 'r', encoding='utf-8') as f:
            template = f.read()
            self.file_handler.write(template)

        indent = "      "
        string_define = [
            "@mytimer(warmup_runs=3, runs=1)",
            "def kernel_generated(basic_stream, bs_result: np.array) -> np.array:",
            f"{indent}basis0 = basic_stream[0]",
            f"{indent}basis1 = basic_stream[1]",
            f"{indent}basis2 = basic_stream[2]",
            f"{indent}basis3 = basic_stream[3]",
            f"{indent}basis4 = basic_stream[4]",
            f"{indent}basis5 = basic_stream[5]",
            f"{indent}basis6 = basic_stream[6]",
            f"{indent}basis7 = basic_stream[7]",
            ""
        ]
        self.file_handler.write("\n".join(string_define))

        self.lower_insts(self.cc_insts, indent)
        self.lower_insts(self.insts, indent)
        self.lower_insts(self.paths_inst, indent)

        string_kernel_end = [
            f"{indent}return bs_result\n",
            "basic_stream = transpose_byte_to_bitstream(input_stream)",
            "bs_result = create_zeros(len(input_stream) + 1)",
            "result, execution_time = kernel_generated(basic_stream, bs_result)",
            "",
        ]
        self.file_handler.write("\n".join(string_kernel_end))
