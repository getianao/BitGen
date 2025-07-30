from .inst import BsInst, BsInstType


class BsSel(BsInst):
    def __init__(self, operands, condition, ret=0):
        super().__init__(BsInstType.SEL, operands, ret)
        self.n_operand = 2
        self.operand1 = operands[0]
        self.operand2 = operands[1]
        self.condition = condition
        self.ret = ret

    # def lower_to_bitstream(self, var_map, indent: str = ""):
    #     ret = self.get_var_name(var_map, self.ret)
    #     operand1 = self.get_var_name(var_map, self.operand1)
    #     code = f"{indent}{ret} = {operand1}"
    #     return code

    def lower_to_python(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        code = f"{indent}{ret} = {operand1}"
        return code

    def lower_to_torch(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        condition = self.get_var_name(var_map, self.condition)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.get_var_name(var_map, self.operand2)
        codes = [
            f"{indent}not_{condition} = torch.bitwise_not({condition})",
            f"{indent}mask_{operand1} = torch.bitwise_and({condition}, {operand1})",
            f"{indent}mask_{operand2} = torch.bitwise_and(not_{condition}, {operand2})",
            f"{indent}{ret} = torch.bitwise_or(mask_{operand1}, mask_{operand2})",
        ]
        code = "\n".join(codes)
        return code

    def get_torch_module(self):
        return "BSSel"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        super().lower_to_cuda(var_map, indent, var_define_map)
        ret = self.get_var_name_cuda(var_map, self.ret)
        condition = self.get_var_name_cuda(var_map, self.condition)
        operand1 = self.get_var_name_cuda(var_map, self.operand1)
        operand2 = self.get_var_name_cuda(var_map, self.operand2)
        codes = [
            f"{indent}{self.ret_type}sel_{ret}_not_{condition} = ~{condition};",
            f"{indent}{self.ret_type}sel_{ret}_mask_{operand1} = {condition} & {operand1};",
            f"{indent}{self.ret_type}sel_{ret}_mask_{operand2} = sel_{ret}_not_{condition} & {operand2};",
            f"{indent}{self.ret_type}{ret} = sel_{ret}_mask_{operand1} | sel_{ret}_mask_{operand2};",
        ]
        code = "\n".join(codes)
        if self.comment != "":
            code += f" // {self.comment}"
        return code
