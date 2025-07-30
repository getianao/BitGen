from .inst import BsInst, BsInstType


class BsGraphBreak(BsInst):
    def __init__(self, operands=None, ret=None):
        super().__init__(BsInstType.GRAPHBREAK, operands, ret)
        self.n_operand = 00
        self.ret = ret
        self.operation = "bs_graph_break"

    def lower_to_bitstream(self, var_map, indent: str = ""):
        code = f"{indent}{self.operation}"
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        return None

    def lower_to_torch(self, var_map, indent: str = ""):
        return None

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        return ""
