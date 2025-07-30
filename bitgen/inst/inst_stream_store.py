from .inst import BsInst, BsInstType


class BsStreamStore(BsInst):
    def __init__(self, operands=None, ret=None):
        super().__init__(BsInstType.STREAMSTORE, operands, ret)
        self.n_operand = 1
        self.operand1 = operands[0]
        self.ret = ret
        self.stream_index = None
        self.operation = "bs_stream_store"
        self.use_or = False
        self.use_atomic_or = False

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
        if self.use_atomic_or:
            code = f"{indent}atomicOr(&{ret}[{self.stream_index}], {operand1});"
        elif self.use_or:
            code = f"{indent}{ret}[{self.stream_index}] |= {operand1};"
        else:
            code = f"{indent}{self.ret_type}{ret}[{self.stream_index}] = {operand1};"
        if self.comment != "":
            code += f" // {self.comment}"
        return code
