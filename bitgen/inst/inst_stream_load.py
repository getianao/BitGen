from .inst import BsInst, BsInstType


class BsStreamLoad(BsInst):
    def __init__(self, operands=None, ret=None):
        super().__init__(BsInstType.STREAMLOAD, operands, ret)
        self.n_operand = 1
        self.operand1 = operands[0]
        self.ret = ret
        self.operation = "bs_stream_load"

    def lower_to_bitstream(self, var_map, indent: str = ""):
        code = f"{indent}{self.operation}"
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        return None

    def lower_to_torch(self, var_map, indent: str = ""):
        return None

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        super().lower_to_cuda(var_map, indent, var_define_map)
        ret = self.get_var_name_cuda(var_map, self.ret)
        operand1 = self.get_var_name_cuda(var_map, self.operand1)
        code = f"{indent}{self.ret_type}{ret} = {operand1}[{self.stream_index}];"
        if self.comment != "":
            code += f" // {self.comment}"
        return code
