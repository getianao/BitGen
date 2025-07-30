from ..inst import BsInstType, BsAssign


class free_intermediate_tensor_pass:
    def __init__(self):
        self.pass_name = "free_intermediate_tensor"
        self.var_none_id = None
        self.var_define_pos = {}
        self.var_last_used = {}

    def define_var(self, var, inst_id):
        self.var_define_pos[var] = inst_id

    def use_var(self, var, inst_id):
        if not isinstance(var, int):
            return
        if var in self.var_last_used:
            if self.var_last_used[var] < inst_id:
                self.var_last_used[var] = inst_id
        else:
            self.var_last_used[var] = inst_id

    def add_to_use(self, inst, inst_id):
        if inst.type == BsInstType.IF:
            for body_1_inst in inst.body_1.body:
                self.add_to_use(body_1_inst, inst_id)
            for body_2_inst in inst.body_2.body:
                self.add_to_use(body_2_inst, inst_id)
            return

        if inst.type == BsInstType.WHILE:
            for body_inst in inst.body.body:
                self.add_to_use(body_inst, inst_id)
            return

        # Add operands to use
        operands = inst.operands
        if isinstance(operands, list):
            for op in operands:
                self.use_var(op, inst_id)
        elif isinstance(operands, int):
            self.use_var(operands, inst_id)
        return

    def add_to_define(self, inst, inst_id):
        ret = inst.ret
        if isinstance(ret, list):
            for r in ret:
                self.define_var(r, inst_id)
        elif isinstance(ret, int):
            self.define_var(ret, inst_id)

    def run(self, insts, var_name_map):
        self.var_none_id = len(var_name_map)
        var_name_map[self.var_none_id] = "None"
        new_insts = insts.copy()
        for inst_id, inst in enumerate(insts):
            self.add_to_use(inst, inst_id)
            self.add_to_define(inst, inst_id)

        update_insts = {}
        for var in self.var_define_pos:
            # print(f"check {self.var_name_map[var]}")
            if var in self.var_last_used:
                inst = BsAssign([self.var_none_id], var)
                update_insts[inst] = self.var_last_used[var]
                # print(f"remove {var_name_map[var]} at {self.var_last_used[var]}" )

        # Sort by inst_pos
        update_insts = {k: v for k, v in sorted(update_insts.items(), key=lambda item: item[1], reverse=True)}
        for inst, inst_id in update_insts.items():
            # print(f"insert {var_name_map[inst.ret]} at {inst_id}")
            new_insts.insert(inst_id + 1, inst)
        return new_insts, var_name_map
