from .inst import BsInst, BsInstType


class BsStr(BsInst):
    def __init__(self, operands=None, ret=0):
        super().__init__(BsInstType.STR, operands, ret)
        self.n_operand = 0
        self.str = operands[0]

    def lower_to_bitstream(self, var_map, indent: str = ""):
        return indent + self.str

    def lower_to_python(self, var_map, indent: str = ""):
        return indent + self.str

    def lower_to_torch(self, var_map, indent: str = ""):
        return indent + self.str

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        code = ""
        if self.str.startswith("LABEL") and self.str.endswith(":"):  # Format label.
            code += f"{self.str}"
        else:
            code += f"{indent}{self.str}"
        if self.comment != "":
            code += f" // {self.comment}"
        return code
