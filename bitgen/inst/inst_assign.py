from .inst import BsInst, BsInstType


class BsAssign(BsInst):
    def __init__(self, operands=None, ret=0):
        super().__init__(BsInstType.ASSIGN, operands, ret)
        self.n_operand = 1
        self.operand1 = operands[0]
        self.ret = ret

    def lower_to_bitstream(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        code = f"{indent}{ret} = {operand1}"
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        code = f"{indent}{ret} = {operand1}"
        return code
    
    def lower_to_torch(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        code = f"{indent}{ret} = {operand1}"
        if self.comment != "":
            code += f" # {self.comment}"
        return code
    
    def get_torch_module(self):
        return "BSAssign"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        super().lower_to_cuda(var_map, indent, var_define_map)
        ret = self.get_var_name_cuda(var_map, self.ret)
        operand1 = self.get_var_name_cuda(var_map, self.operand1)
        code = f"{indent}{self.ret_type}{ret} = {operand1};"
        if self.comment != "":
            code += f" // {self.comment}"
        return code
    
    
