from .inst import BsInst, BsInstType
from itertools import chain


class BsBlock(BsInst):

    def __init__(self, operands=None, ret=0, name: str = ""):
        super().__init__(BsInstType.BLOCK, operands, ret)
        self.n_operand = 1
        self.name = str(name)
        self.body = operands[0]
        self.ret = ret

    def lower_to_bitstream(self, var_map, indent: str = ""):
        codes = []
        codes.append(f"{indent}{{")
        body_insts = list(chain(self.body))
        for inst in body_insts:
            codes.append(inst.lower_to_python(var_map, indent + "    "))
        codes.append(f"{indent}}}")
        code = "\n".join(codes)
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        codes = []
        body_insts = list(chain(self.body))
        indent += "    "
        # codes.append(f"{indent}# block {self.name} start")
        for inst in body_insts:
            codes.append(inst.lower_to_python(var_map, indent))
        # codes.append(f"{indent}# block {self.name} end")
        code = "\n".join(codes)
        return code
    
    def lower_to_torch(self, var_map, indent: str = ""):
        codes = []
        body_insts = list(chain(self.body))
        indent += "    "
        # codes.append(f"{indent}# block {self.name} start")
        for inst in body_insts:
            codes.append(inst.lower_to_torch(var_map, indent))
        # codes.append(f"{indent}# block {self.name} end")
        code = "\n".join(codes)
        return code
    
    def get_torch_module(self):
        return "BSBlock"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        var_define_map_bk = var_define_map.copy()
        # super().lower_to_cuda(var_map, indent, var_define_map)
        codes = []
        body_insts = list(chain(self.body))
        for inst in body_insts:
            codes.append(inst.lower_to_cuda(var_map, indent, var_define_map))
        var_define_map.clear()
        var_define_map.update(var_define_map_bk)
        code = "\n".join(codes)
        return code

    def __str_body__(self):
        block_string = ""
        for idx, inst in enumerate(self.body):
            block_string += str(inst)
            if idx != len(self.body) - 1:
                block_string += ", "
        block_string = "[" + block_string + "]"
        return block_string

    def __str__(self):
        # body_name = ""
        # if self.name != None and len(self.name) != 0:
        #     body_name = f"\"{self.name}\""

        return f"Block"+ self.__str_body__()

    def __repr__(self):
        return self.__str__();
