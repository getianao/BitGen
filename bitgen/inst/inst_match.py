from .inst import BsInst, BsInstType
from ..cc_parser import Parser


class BsMatch(BsInst):
    def __init__(self, operands=None, ret=0):
        super().__init__(BsInstType.MATCH, operands, ret)
        self.n_operand = 2
        self.operand1 = operands[0]
        self.operand2 = operands[1]
        self.ret = ret

    def lower_to_bitstream(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.operand2
        operation = "bs_match"
        code = f'{indent}{ret} = {operation}({operand1}, "{operand2}")'
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.operand2
        operation = "bs_match"
        code = f'{indent}{ret} = {operation}({operand1}, "{operand2}")'
        return code
    
    def lower_to_torch(self, var_map, indent: str = ""):
        ret = self.get_var_name(var_map, self.ret)
        operand1 = self.get_var_name(var_map, self.operand1)
        operand2 = self.operand2
        operation = "bs_match"
        code = f'{indent}{ret} = {operation}({self.operation}, "{operand2}")'
        if self.comment != "":
            code += f" # {self.comment}"
        return code
    
    def get_torch_module(self):
        return "BSMatch"
    

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        super().lower_to_cuda(var_map, indent, var_define_map)
        ret = self.get_var_name_cuda(var_map, self.ret)
        operand1 = self.get_var_name_cuda(var_map, self.operand1)
        operand2 = self.operand2
        operation = "bs_match"

        codes = []
        parser = Parser()
        parser.parse_symbol_set(operand2)
        symbolset = [hex(num) for num in parser.column]
        symbolset = "{" + ", ".join(symbolset) + "}"
        codes.append(f"{indent}const {self.ret_type}symbolset_{ret}[] = {symbolset};")
        codes.append(      
            f"{indent}{self.ret_type}{ret} = {operation}({operand1}, ((bit_id + 32U >= n_char && bit_id < n_char) ? (n_char - bit_id) : 32), symbolset_{ret});"
        )
        code = "\n".join(codes)
        return code
