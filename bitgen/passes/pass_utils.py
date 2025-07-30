import networkx as nx

from ..inst import BsInstType


def remove_empty_insts(insts):
    return [inst for inst in insts if inst != None]


def update_insts(insts, insert_insts: dict):
    # Sort by inst_pos from last to first
    insert_insts = {
        k: v
        for k, v in sorted(
            insert_insts.items(),
            key=lambda item: (-item[1], -(list(insert_insts.keys()).index(item[0]))),
        )
    }
    for inst, inst_id in insert_insts.items():
        # print(f"insert {self.var_name_map[inst.ret]} at {inst_id}")
        insts.insert(inst_id, inst)
    return insts


inst_name_map = {
    BsInstType.AND: "and",
    BsInstType.OR: "or",
    BsInstType.NOT: "not",
    BsInstType.XOR: "xor",
    BsInstType.ADD: "add",
    BsInstType.MATCHSTAR: "mstar",
    BsInstType.SCANTHRU: "sthru",
    BsInstType.ASSIGN: "ass",
    BsInstType.SEL: "sel",
    BsInstType.IF: "if",
    BsInstType.WHILE: "while",
    BsInstType.ADVANCE: "adv",
    BsInstType.TERNARY: "ternary",
    BsInstType.MATCH: "match",
    BsInstType.CALL: "call",
    BsInstType.BLOCK: "block",
    BsInstType.STATE: "state",
    BsInstType.GRAPHBREAK: "graph_break",
    BsInstType.STREAMLOAD: "stream_load",
    BsInstType.STREAMSTORE: "stream_store",
    BsInstType.STREAMDEFINE: "stream_define",
    BsInstType.SCANTHRUSTREAM: "stream_scanthru",
    BsInstType.STR: "str",
}


def build_graph(insts, var_name_map):
    G = nx.DiGraph()

    define_pos = {}  # var_id -> inst_id
    edge_labels = {}

    for inst_id, inst in enumerate(insts):
        # print(inst.lower_to_torch(var_name_map))
        if inst.type == BsInstType.STR:
            continue
        if isinstance(inst.ret, list):
            assert len(inst.ret) == 1
            define_pos[inst.ret[0]] = inst_id
        else:
            define_pos[inst.ret] = inst_id

        G.add_node(inst_id, inst=inst, label=f"{inst_id}\n{inst_name_map[inst.type]}")

        if inst.type == BsInstType.WHILE:
            operand = inst.condition
            if operand in define_pos:
                # print(f"add edge {define_pos[operand]} -> {inst_id}")
                # if define_pos[operand] == inst_id:
                #     print(inst)
                #     raise Exception(f"Self loop detected: operand:{var_name_map[operand]} define at {inst_id}")
                G.add_edge(
                    define_pos[operand],
                    inst_id,
                    label=f"{var_name_map[operand]}",
                )
        else:
            for operand_id in range(inst.n_operand - 1, -1, -1):
                if inst.type == BsInstType.ADVANCE and operand_id == 1:
                    continue
                operand = inst.operands[operand_id]
                if operand in define_pos:
                    # Self-loop
                    if define_pos[operand] == inst_id:
                        continue
                    # To reduce edges, those from non-AND nodes and between reachable nodes are omitted.
                    if nx.has_path(G, define_pos[operand], inst_id):
                        # print(f"Path exists between {define_pos[operand]} and {inst_id}")
                        if inst.type != BsInstType.AND:
                            continue

                    G.add_edge(
                        define_pos[operand],
                        inst_id,
                        label=f"{var_name_map[operand]}",
                    )
    return G


def break_insts(insts):
    insts_break_group = []
    insts_break = []
    for inst_id, inst in enumerate(insts):
        if inst.type == BsInstType.GRAPHBREAK:
            if len(insts_break) > 0:
                insts_break_group.append(insts_break)
                insts_break = []
        else:
            insts_break.append(inst)
    if len(insts_break) > 0:
        insts_break_group.append(insts_break)
    return insts_break_group


def add_new_var(var_name_map, var_name):
    # Found existing var_id
    if var_name in var_name_map.values():
        for var_id, name in var_name_map.items():
            if name == var_name:
                return var_id
    # Add new var_id
    new_var_id = len(var_name_map)
    var_name_map[new_var_id] = var_name
    return new_var_id


def get_var_by_name(var_name, var_name_map):
    for var in var_name_map:
        if var_name_map[var] == var_name:
            return var
    return None
