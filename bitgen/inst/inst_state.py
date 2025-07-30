from .inst import BsInst, BsInstType
from .inst_block import BsBlock
from itertools import chain


class BsState(BsBlock):

    # def lower_to_bitstream(self, var_map, indent: str = ""):
    #     ret = self.get_var_name(var_map, self.ret)
    #     operand1 = self.get_var_name(var_map, self.operand1)
    #     operand2 = self.operand2
    #     operation = "bs_match"
    #     code = f'{indent}{ret} = {operation}({operand1}, "{operand2}")'
    #     return code

    def lower_to_python(self, var_map, indent: str = ""):
        codes = []
        body_insts = list(chain(self.body))
        codes.append(f"{indent}# State {self.name}")
        for inst in body_insts:
            codes.append(inst.lower_to_python(var_map, indent))
        # codes.append(f"{indent}# State {self.name} end")
        code = "\n".join(codes)
        return code
    
    def lower_to_torch(self, var_map, indent: str = ""):
        codes = []
        body_insts = list(chain(self.body))
        codes.append(f"{indent}# State {self.name}")
        for inst in body_insts:
            codes.append(inst.lower_to_torch(var_map, indent))
        # codes.append(f"{indent}# State {self.name} end")
        code = "\n".join(codes)
        return code
    
    def get_torch_module(self):
        return "BSState"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        super().lower_to_cuda(var_map, indent, var_define_map)
        codes = []
        body_insts = list(chain(self.body))
        codes.append(f"{indent}// State {self.name}")
        for inst in body_insts:
            codes.append(inst.lower_to_cuda(var_map, indent, var_define_map))
        # codes.append(f"{indent}# State {self.name} end")
        code = "\n".join(codes)
        return code

    def __str__(self):
        return f"State({self.name})"

    def __repr__(self):
        return self.__str__()
