from .inst import BsInst, BsInstType
from .inst_block import BsBlock
from itertools import chain


class BsIf(BsInst):

    def __init__(self, operands, condition, ret=None):
        super().__init__(BsInstType.IF, operands, ret)
        self.n_operand = 1
        self.condition = condition
        self.body_1 = operands[0]
        self.body_2 = operands[1]
        self.ret = ret  # The return variables come from 2 body_blocks

    def lower_to_python(self, var_map, indent: str = ""):
        codes = []
        # super().lower_to_cuda(var_map, indent, var_define_map)
        condition = self.get_var_name(var_map, self.condition)
        codes.append(f"{indent}if ({condition}.any()):")
        codes.append(self.body_1.lower_to_python(var_map, indent))
        if self.body_2 != None:
            codes.append(f"{indent}else:")
            codes.append(self.body_2.lower_to_python(var_map, indent))
        code = "\n".join(codes)
        return code

    def lower_to_torch(self, var_map, indent: str = ""):
        codes = []
        # super().lower_to_cuda(var_map, indent, var_define_map)
        condition = self.get_var_name(var_map, self.condition)
        codes.append(f"{indent}if (torch.any({condition}!=0)):")
        codes.append(self.body_1.lower_to_torch(var_map, indent))
        if self.body_2 != None:
            codes.append(f"{indent}else:")
            assert self.body_2.type == BsInstType.BLOCK
            assert len(self.body_2.body) == 1
            assert self.body_2.body[0].type == BsInstType.ASSIGN
            assert var_map[self.body_2.body[0].operand1] == "0"
            body_2 = self.body_2.deepcopy()
            body_2.body[0].operand1 = condition  # Reuse zero tensor
            codes.append(body_2.lower_to_torch(var_map, indent))
        code = "\n".join(codes)
        return code

    def get_torch_module(self):
        return "BSIf"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        codes = []

        # super().lower_to_cuda(var_map, indent, var_define_map)
        indent2 = indent + "    "
        condition = self.get_var_name_cuda(var_map, self.condition)
        codes.append(f"{indent}if ({condition}) {{")
        codes.append(
            self.body_1.lower_to_cuda(var_map, indent2, var_define_map)
        )
        if self.body_2 != None:
            codes.append(f"{indent}else {{")
            codes.append(self.body_2.lower_to_cuda(var_map, indent2, var_define_map))
            codes.append(f"{indent}}}")
        codes.append(f"{indent}}}")
        code = "\n".join(codes)
        return code

    def __str__(self):
        if self.body_2 == None:
            return f"If[{self.body_1}]"
        return f"If[{self.body_1}, {self.body_2}]"

    def __repr__(self):
        return self.__str__()
