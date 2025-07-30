from .inst import BsInst, BsInstType


class BsAdvance(BsInst):
    def __init__(self, operands=None, ret=0):
        super().__init__(BsInstType.ADVANCE, operands, ret)
        self.operation = "BSAdvanceRightFunctionSync"
        self.n_operand = 2
        self.operand1 = operands[0]
        self.operand2 = int(operands[1])
        self.dymatic_offset = None
        self.ret = ret
        self.shared_adv_mem_offset = 0

    def lower_to_bitstream(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.operand2
        code = f"{indent}{ret} = {self.operation}({operand1}, {operand2})"
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.operand2
        code = f"{indent}{ret} = {self.operation}({operand1}, {operand2})"
        return code

    def lower_to_torch(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.operand2
        code = f"{indent}{ret} = {self.operation}({operand1}, {operand2})"
        if self.comment != "":
            code += f" # {self.comment}"
        return code

    def get_torch_module(self):
        return "BSAdvance"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        super().lower_to_cuda(var_map, indent, var_define_map)
        ret = self.get_var_name_cuda(var_map, self.ret)
        operand1 = self.get_var_name_cuda(var_map, self.operand1)
        if self.dymatic_offset:
            operand2 = f"{self.dymatic_offset} + {self.operand2}"
        else:
            operand2 = self.operand2
        code = f"{indent}{self.ret_type}{ret} = {self.operation}({operand1}, {operand2}, advance_memory + {self.shared_adv_mem_offset});"
        if self.comment != "":
            code += f" // {self.comment}"
        return code
