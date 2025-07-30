from .inst import BsInst, BsInstType


class BsXor(BsInst):
    def __init__(self, operands=None, ret=0):
        super().__init__(BsInstType.XOR, operands, ret)
        self.n_operand = 2
        self.operand1 = operands[0]
        self.operand2 = operands[1]
        self.ret = ret

    def lower_to_bitstream(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.get_var_name(var_map, self.operand2)
        operation = "bs_xor"
        code = f"{indent}{ret} = {operation}({operand1}, {operand2})"
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.get_var_name(var_map, self.operand2)
        operation = "bs_xor"
        code = f"{indent}{ret} = {operation}({operand1}, {operand2})"
        return code

    def lower_to_torch(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.get_var_name(var_map, self.operand2)
        operation = "bs_xor"
        code = f"{indent}{ret} = {self.operation}({operand1}, {operand2})"
        if self.comment != "":
            code += f" # {self.comment}"
        return code

    def get_torch_module(self):
        return "BSXor"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        super().lower_to_cuda(var_map, indent, var_define_map)
        ret = self.get_var_name_cuda(var_map, self.ret)
        operand1 = self.get_var_name_cuda(var_map, self.operand1)
        operand2 = self.get_var_name_cuda(var_map, self.operand2)
        operation = "bs_xor"
        code = f"{indent}{self.ret_type}{ret} = {operation}({operand1}, {operand2});"
        if self.comment != "":
            code += f" // {self.comment}"
        return code
