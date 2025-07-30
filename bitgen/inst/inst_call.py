from .inst import BsInst, BsInstType


class BsCall(BsInst):
    def __init__(self, operands=None, ret=0):
        super().__init__(BsInstType.CALL, operands, ret)
        self.n_operand = 0
        self.operand1 = operands[0]  # operation
        self.operand2 = operands[1:]  # params
        self.ret = ret

    def lower_to_bitstream(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operation = self.operand1
        params = self.operand2
        code = f"{indent}{ret} = {operation}({', '.join(params)})"
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operation = self.operand1
        params = self.operand2
        code = f"{indent}{ret} = {operation}({', '.join(params)})"
        return code

    def lower_to_torch(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        self.operation = self.operand1
        params = self.operand2
        if ret == None:
            code = f"{indent}{self.operation}({', '.join(params)})"
        else:
            code = f"{indent}{ret} = {self.operation}({', '.join(params)})"
        if self.comment != "":
            code += f" # {self.comment}"
        return code

    def get_torch_module(self):
        return "BSCall"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        super().lower_to_cuda(var_map, indent, var_define_map)
        ret = self.get_var_name_cuda(var_map, self.ret)
        operation = self.operand1
        params = self.operand2
        if operation == "create_ones":
            code = f"{indent}{self.ret_type}{ret} = 0xffffffff;"
        else:
            code = f"{indent}{self.ret_type}{ret} = {params}({', '.join(params)};)"
        if self.comment != "":
            code += f" // {self.comment}"
        return code
