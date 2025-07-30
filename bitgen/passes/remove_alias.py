from ..inst import BsInstType, BsAssign, BsAdvance


class remove_alias_pass:
    def __init__(self,):
        self.pass_name = "remove_alias"
        self.define_pos = {} # var_id -> inst_id
        self.alias = {} # alias -> defined_var_id

    def run(self, insts, var_name_map):
        new_insts = insts.copy()
        for inst_id, inst in enumerate(insts):
            # Update define postion
            if inst.type == BsInstType.WHILE:
                continue
            self.define_pos[inst.ret] = inst_id
            if inst.type == BsInstType.ASSIGN:
                ret_name = var_name_map[inst.ret]
                if ret_name == "output.matches" or ret_name.startswith(  # result
                    "basis"
                ):  # basic stream
                    continue
                # while
                if ret_name.startswith("test") or ret_name.startswith("accum"):
                    operand = inst.operand1
                    if operand in self.alias:
                        new_insts[inst_id].operand1 = self.alias[operand]
                        new_insts[inst_id].operands[0] = self.alias[operand]
                    continue
                if inst.operand1 in self.alias and self.alias != None:
                    self.alias[inst.ret] = self.alias[inst.operand1]
                else:
                    self.alias[inst.ret] = inst.operand1
                new_insts[inst_id] = None
            else:
                # Redefine the alas var
                if inst.ret in self.alias:
                    self.alias.pop(inst.ret)
                # Replace the operand with defined var
                for operand_id in range(inst.n_operand):
                    operand = inst.operands[operand_id]
                    if operand in self.alias:
                        new_insts[inst_id].operands[operand_id] = self.alias[operand]
                        # TODO(tge): Fix operand1 usage
                        if operand_id == 0:
                            new_insts[inst_id].operand1 = self.alias[operand]
                            new_insts[inst_id].operands[0] = self.alias[operand]
                        elif operand_id == 1:
                            new_insts[inst_id].operand2 = self.alias[operand]
                            new_insts[inst_id].operands[1] = self.alias[operand]
        new_insts = [inst for inst in new_insts if inst != None]
        return new_insts, var_name_map
