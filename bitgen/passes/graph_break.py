from itertools import chain

from ..inst import *
from ..passes.pass_utils import update_insts, break_insts, add_new_var
from ..log import MyLogger

class graph_break_pass:
    def __init__(self, graph_break_type=0):
        self.pass_name = "graph_break"
        self.graph_break_type = graph_break_type

    def check_insts_exsist(self, insert_insts, operand, ret, subgraph_id):
        for inst, pos_id in insert_insts.items():
            if inst.operands[0] == operand and inst.ret == ret and pos_id == subgraph_id:
                return True
        return False

    def print_define_map(self, var_define_map, var_name_map):
        for var_id, inst_id in var_define_map.items():
            print(f"{var_name_map.get(var_id, var_id)} define at: {inst_id}")

    def print_var_define_map_subgraphs(self, var_define_map_subgraphs, var_name_map):
        for subgraph_id, var_define_map in enumerate(var_define_map_subgraphs):
            for var_id, inst_id in var_define_map.items():
                print(
                    f"{var_name_map.get(var_id, var_id)} define at: inst {inst_id} in subgraph {subgraph_id}"
                )

    def get_defines(
        self, var_define_map_break_group, var_id, var_stream_id, var_name_map
    ):
        inst_stream_stores = []
        subgraph_ids = []
        for subgraph_id, var_define_map in enumerate(var_define_map_break_group):
            if var_id in var_define_map:
                inst_stream_store = BsStreamStore([var_id], var_stream_id)
                inst_stream_store.stream_index = "idx"
                inst_stream_stores.append(inst_stream_store)
                subgraph_ids.append(subgraph_id)
        if len(inst_stream_stores) == 0:
            raise Exception(
                f"Cannot find define for var_id: {var_id}, {var_name_map.get(var_id, var_id)}"
            )
        return inst_stream_stores, subgraph_ids

    def search_var_name(self, var_name_map, var_name):
        for var_id, name in var_name_map.items():
            if name == var_name:
                return var_id
        return None

    # Return stream var
    def search_var_define_across_subgraph(
        self,
        insts_stream_store_inserts,
        insts_stream_load_inserts,
        insts_stream_define_inserts,
        var_define_map_break_group,
        insts_break_id,
        var_name_map,
        operand_id,
        inst_id,
    ):
        if isinstance(operand_id, str):
            return None

        MyLogger.debug(f"[graph_break] check operand: {var_name_map.get(operand_id, operand_id)}.")
        if (operand_id in var_define_map_break_group[insts_break_id]):
            MyLogger.debug(f"[graph_break] define at: { var_define_map_break_group[insts_break_id][operand_id]} , use at: {inst_id}")
        if (
            operand_id not in var_define_map_break_group[insts_break_id]
            or inst_id < var_define_map_break_group[insts_break_id][operand_id]
        ):
            MyLogger.debug(f"[graph_break] Found undefined operand: {operand_id}, {var_name_map.get(operand_id, operand_id)}")

            # If stream for the operand is already defined, skip store and define.
            var_name = var_name_map[operand_id]
            var_stream_id = self.search_var_name(var_name_map, var_name + "_stream")
            if var_stream_id is None:
                # Add stream define inst in the begining
                var_stream_id = add_new_var(var_name_map, var_name + "_stream")
                var_tmp_streams_id = add_new_var(var_name_map, "tmp_streams")
                inst_steam_define = BsStreamDefine([var_tmp_streams_id], var_stream_id)
                insts_stream_define_inserts.append(inst_steam_define)
                MyLogger.debug(f"[graph_break] add stream define inst")
            # Add stream store inst in all subgraphs that define the operand
            try:
                inst_stream_stores, subgraph_ids = self.get_defines(
                    var_define_map_break_group, operand_id, var_stream_id, var_name_map
                )
                for inst_stream_store, subgraph_id in zip(
                    inst_stream_stores, subgraph_ids
                ):
                    if not self.check_insts_exsist(
                        insts_stream_store_inserts,
                        inst_stream_store.operands[0],
                        inst_stream_store.ret,
                        subgraph_id,
                    ):
                        insts_stream_store_inserts[inst_stream_store] = subgraph_id
                        MyLogger.debug(
                            f"[graph_break] add stream store inst in subgraph {subgraph_id}"
                        )
            except Exception as e:
                pass
            # Add stream load inst in the current subgraph
            # Skip redundant stream load inst in the same subgraph
            for (
                already_insert_inst,
                already_insert_subgraph_id,
            ) in insts_stream_load_inserts.items():
                if (
                    already_insert_inst.operands[0] == var_stream_id
                    and already_insert_subgraph_id == insts_break_id
                ):
                    return var_stream_id
            inst_stream_load = BsStreamLoad([var_stream_id], operand_id)
            inst_stream_load.stream_index = "idx"
            insts_stream_load_inserts[inst_stream_load] = insts_break_id
            MyLogger.debug(f"[graph_break] add stream load inst in subgraph {insts_break_id}")
            return var_stream_id
        return None

    def add_var_define(
        self, var_define_map_subgraphs, var_define_map_subgraph, inst, inst_id
    ):
        if isinstance(inst.ret, list):
            assert len(inst.ret) == 1
            ret = inst.ret[0]
        else:
            ret = inst.ret
        # # Check previous subgraph
        # for other_subgraph in var_define_map_subgraphs:
        #     if ret in other_subgraph:
        #         return other_subgraph[ret]
        # Check current subgraph
        if ret in var_define_map_subgraph:
            return var_define_map_subgraph[ret]
        # First position of defination.
        var_define_map_subgraph[ret] = inst_id
        return inst_id

    def get_inst_while_in_subgraph(self, insts_subgraph):
        for inst in insts_subgraph:
            if inst.type == BsInstType.WHILE:
                return inst
        return None

    def insert_if_inst_for_stream_store(self, insts):
        inst_stream_store_start = None
        for inst_id, inst in enumerate(insts):
            if inst.type == BsInstType.STREAMSTORE:
                inst_stream_store_start = inst_id
                break
        if inst_stream_store_start is None:
            # print("No stream store inst in the subgraph:", insts)
            return insts
        use_or = False
        for inst_id in range(0, inst_stream_store_start):
            inst = insts[inst_id]
            if inst.type == BsInstType.ADVANCE:
                use_or = True
                break
        for inst_stream_store_id in range(inst_stream_store_start, len(insts)):
            inst = insts[inst_stream_store_id]
            if inst.type != BsInstType.STREAMSTORE:
                raise Exception(f"Non-stream store inst at the end of insts")
            if use_or:
                insts[inst_stream_store_start].use_or = True
        # print("length of stream store", len(insts) - inst_stream_store_start)
        if insts[inst_stream_store_start - 1].type == BsInstType.IF:
            # Merge if
            insts[inst_stream_store_start - 1].body_1.body += insts[
                inst_stream_store_start:
            ]
            insts = insts[:inst_stream_store_start]
        else:
            inst_block = BsBlock([insts[inst_stream_store_start:]])
            inst_if = BsIf([inst_block, None], condition="idx < n_unit_basic")
            insts = insts[:inst_stream_store_start]
            insts.append(inst_if)
        return insts

    def run_in_while_loop(self, insts, var_name_map):
        new_insts = insts.copy()
        insert_insts = {}  # inst -> pos_id

        # Insert graph break
        # TODO(tge): graph_break_all in loop
        if self.graph_break_type and 0:
            for inst_id, inst in enumerate(insts):
                if inst.type not in [
                    BsInstType.STREAMSTORE,
                    BsInstType.STREAMLOAD,
                    BsInstType.STREAMDEFINE,
                    BsInstType.IF,
                ]:
                    insert_insts[BsGraphBreak()] = inst_id + 1
        else:
            for inst_id, inst in enumerate(insts):
                if inst.type in [BsInstType.WHILE, BsInstType.SCANTHRU]:
                    insert_insts[BsGraphBreak()] = inst_id
                    insert_insts[BsGraphBreak()] = inst_id + 1
        update_insts(new_insts, insert_insts)

        # Group insts by graph break
        insts_subgraphs = break_insts(new_insts)

        # Generate var define map for each subgraph
        var_define_map_subgraphs = []
        for insts_subgraph_id, insts_subgraph in enumerate(insts_subgraphs):
            var_define_map_subgraph = {}  # var_id -> inst_id
            for inst_id, inst in enumerate(insts_subgraph):
                if inst.type == BsInstType.WHILE:
                    for inst_body_id, inst_body in enumerate(
                        list(chain(inst.body.body))
                    ):
                        self.add_var_define(
                            var_define_map_subgraphs,
                            var_define_map_subgraph,
                            inst_body,
                            inst_body_id,
                        )
                else:
                    self.add_var_define(
                        var_define_map_subgraphs, var_define_map_subgraph, inst, inst_id
                    )
            var_define_map_subgraphs.append(var_define_map_subgraph)
        # self.print_var_define_map_subgraphs(var_define_map_subgraphs, var_name_map)
        # Search var define and usage across different subgraph
        insts_stream_define_inserts = [] # inst
        insts_stream_store_inserts = {}  # inst -> group_id
        insts_stream_load_inserts = {}  # inst -> group_id

        for insts_subgraph_id in reversed(range(len(insts_subgraphs))):
            insts_subgraph = insts_subgraphs[insts_subgraph_id]
            MyLogger.debug(f"[graph_break] check subgraph {insts_subgraph_id}")
            for inst_id, inst in enumerate(insts_subgraph):
                if inst.type in [BsInstType.STREAMLOAD, BsInstType.IF]:
                    continue
                if inst.type == BsInstType.WHILE:
                    condition_stream_name = var_name_map[inst.condition]+"_stream"
                    condition_stream_id = self.search_var_name(var_name_map, condition_stream_name)
                    if condition_stream_id is None:
                        condition_stream_id = add_new_var(var_name_map, condition_stream_name)
                        var_tmp_streams_id = add_new_var(var_name_map, "tmp_streams")
                        inst_stream_define = BsStreamDefine([var_tmp_streams_id], condition_stream_id)
                        insts_stream_define_inserts.append(inst_stream_define)
                    inst.condition = condition_stream_id
                    for inst_body_id, inst_body in enumerate(inst.body.body):
                        assert inst_body.type != BsInstType.WHILE
                        for operand_idx in range(inst_body.n_operand):
                            operand_id = inst_body.operands[operand_idx]
                            self.search_var_define_across_subgraph(
                                insts_stream_store_inserts,
                                insts_stream_load_inserts,
                                insts_stream_define_inserts,
                                var_define_map_subgraphs,
                                insts_subgraph_id,
                                var_name_map,
                                operand_id,
                                inst_body_id,
                            )
                    continue
                for operand_idx in range(inst.n_operand):
                    operand_id = inst.operands[operand_idx]
                    self.search_var_define_across_subgraph(
                        insts_stream_store_inserts,
                        insts_stream_load_inserts,
                        insts_stream_define_inserts,
                        var_define_map_subgraphs,
                        insts_subgraph_id,
                        var_name_map,
                        operand_id,
                        inst_id,
                    )
        # Insert stream store insts.
        for inst_stream_store in insts_stream_store_inserts:
            insts_subgraph_id = insts_stream_store_inserts[inst_stream_store]
            insts_subgraph = insts_subgraphs[insts_subgraph_id]
            # When the subgraph is a while loop, insert the stream store in the loop body.
            inst_while = self.get_inst_while_in_subgraph(insts_subgraph)
            # Check useless store if load from the same stream and not modify
            operator = inst_stream_store.operands[0]
            is_load_from_same_stream = False
            is_not_modify = True
            for inst_check in insts_subgraph:
                if (
                    inst_check.type == BsInstType.STREAMLOAD
                    and inst_check.ret == operator
                ):
                    if inst_check.operands[0] == inst_stream_store.ret:
                        is_load_from_same_stream = True
                        break
            if is_load_from_same_stream:
                for inst_check in insts_subgraph:
                    for operand_check_id in inst_check.operands:
                        if operator == inst_stream_store.ret:
                            is_not_modify = False
                            break
            if is_load_from_same_stream and is_not_modify:
                continue
            if inst_while is not None:
                insts_subgraph[0].body.body.append(inst_stream_store)
                ret_name = var_name_map[inst_stream_store.ret]
                if ret_name.startswith("test") or ret_name.startswith("accum"):
                    condition_stream_name = var_name_map[inst_stream_store.ret]
                    condition_stream_name_next = condition_stream_name + "_next"
                    condition_stream_next = add_new_var(
                        var_name_map, condition_stream_name_next
                    )
                    inst_stream_store.ret = condition_stream_next
                    var_tmp_streams_id = add_new_var(var_name_map, "tmp_streams")
                    inst_stream_define = BsStreamDefine(
                        [var_tmp_streams_id], condition_stream_next
                    )
                    insts_stream_define_inserts.append(inst_stream_define)
                    inst_while.cuda_swap_pointer.append(
                        f"swap_pointer(&{condition_stream_name}, &{condition_stream_name_next});"
                    )
                # insts_subgraph.append(inst_stream_store)
            else:
                insts_subgraph.append(inst_stream_store)

        # Insert stream load insts.
        for inst_stream_load in insts_stream_load_inserts:
            insts_subgraph_id = insts_stream_load_inserts[inst_stream_load]
            insts_subgraph = insts_subgraphs[insts_subgraph_id]

            # When the subgraph is a while loop, insert the stream load in the loop body.
            inst_while = self.get_inst_while_in_subgraph(insts_subgraph)
            if inst_while is not None:
                inst_while.body.body.insert(0, inst_stream_load)
            else:
                insts_subgraph.insert(0, inst_stream_load)

        # Insert stream define insts.
        for inst_stream_define in insts_stream_define_inserts:
            insts_subgraphs[0].insert(0, inst_stream_define)

        # Set memory boundary for stream store insts.
        for subgraph_id in range(len(insts_subgraphs)):
            inst_stream_while = self.get_inst_while_in_subgraph(
                insts_subgraphs[subgraph_id]
            )
            if inst_stream_while is not None:
                continue
                raise Exception("While loop can't be nested in while loop")
            else:
                insts_subgraphs[subgraph_id] = self.insert_if_inst_for_stream_store(
                    insts_subgraphs[subgraph_id]
                )

        for insts_subgraph in insts_subgraphs:
            inst_whole = self.get_inst_while_in_subgraph(insts_subgraph)
            if inst_whole is not None:
                insts_while_body = inst_whole.body.body
                self.run_in_while_loop(insts_while_body, var_name_map)

        new_insts = []
        for insts_subgraph in insts_subgraphs:
            new_insts += insts_subgraph
            new_insts.append(BsGraphBreak())

        # print ("new_insts", new_insts)
        return new_insts, var_name_map

    def run(self, insts, var_name_map):
        new_insts = insts.copy()
        insert_insts = {}  # inst -> pos_id

        # Insert graph break
        if self.graph_break_type == 2:
            # Baseline. Fuse for and, or, not.
            for inst_id, inst in enumerate(insts):
                if inst.type in [
                    BsInstType.WHILE,
                    BsInstType.SCANTHRU,
                    BsInstType.ADVANCE,
                ]:
                    insert_insts[BsGraphBreak()] = inst_id
                    insert_insts[BsGraphBreak()] = inst_id + 1
        elif self.graph_break_type == 1:
            # A very bad baseline. One inst one loop.
            for inst_id, inst in enumerate(insts):
                if inst.type not in [
                    BsInstType.STREAMSTORE,
                    BsInstType.STREAMLOAD,
                    BsInstType.STREAMDEFINE,
                ]:
                    insert_insts[BsGraphBreak()] = inst_id
        elif self.graph_break_type == 0:
            # Only fuse shift inst.
            for inst_id, inst in enumerate(insts):
                if inst.type in [BsInstType.WHILE, BsInstType.SCANTHRU]:
                    insert_insts[BsGraphBreak()] = inst_id
                    insert_insts[BsGraphBreak()] = inst_id + 1
        elif self.graph_break_type == -1:
            # Fuse all
            # Add if inst for the last result_stream store.
            last_inst = insts[-1]
            if last_inst.type != BsInstType.STREAMSTORE:
                raise Exception("Last inst is not stream store")
            inst_block = BsBlock([[last_inst]])
            inst_if = BsIf([inst_block, None], condition="idx < n_unit_basic")
            new_insts[-1] = inst_if
            return new_insts, var_name_map
        else:
            raise Exception(f"Unknown graph break type: {self.graph_break_type}")
        update_insts(new_insts, insert_insts)

        # Group insts by graph break
        insts_subgraphs = break_insts(new_insts)

        # Generate var define map for each subgraph
        var_define_map_subgraphs = []
        for insts_subgraph_id, insts_subgraph in enumerate(insts_subgraphs):
            var_define_map_subgraph = {}  # var_id -> inst_id
            for inst_id, inst in enumerate(insts_subgraph):
                if inst.type == BsInstType.WHILE:
                    for inst_body_id, inst_body in enumerate(
                        list(chain(inst.body.body))
                    ):
                        self.add_var_define(
                            var_define_map_subgraphs,
                            var_define_map_subgraph,
                            inst_body,
                            inst_body_id,
                        )
                else:
                    self.add_var_define(
                        var_define_map_subgraphs, var_define_map_subgraph, inst, inst_id
                    )
            var_define_map_subgraphs.append(var_define_map_subgraph)

        # for subgraph_id, var_define_map_subgraph in enumerate(var_define_map_subgraphs):
        #     MyLogger.debug(f"[graph_break] print subgraph {subgraph_id}")
        #     self.print_define_map(var_define_map_subgraph, var_name_map)

        # Search var define and usage across different subgraph
        insts_stream_define_inserts = [] # inst
        insts_stream_store_inserts = {}  # inst -> group_id
        insts_stream_load_inserts = {}  # inst -> group_id

        for insts_subgraph_id in reversed(range(len(insts_subgraphs))):
            insts_subgraph = insts_subgraphs[insts_subgraph_id]
            MyLogger.debug(f"[graph_break] check subgraph {insts_subgraph_id}")
            for inst_id, inst in enumerate(insts_subgraph):
                if inst.type in [BsInstType.STREAMLOAD, BsInstType.IF]:
                    continue
                if inst.type == BsInstType.WHILE:
                    condition_stream = self.search_var_define_across_subgraph(
                        insts_stream_store_inserts,
                        insts_stream_load_inserts,
                        insts_stream_define_inserts,
                        var_define_map_subgraphs,
                        insts_subgraph_id,
                        var_name_map,
                        inst.condition,
                        inst_id,
                    )
                    if condition_stream is not None:
                        inst.condition = condition_stream

                    for inst_body_id, inst_body in enumerate(
                        list(chain(inst.body.body))
                    ):
                        if inst_body.type == BsInstType.WHILE:
                            #     condition2_name = var_name_map[inst_body.condition]
                            #     condition2_stream = self.search_var_name(var_name_map, condition2_name + "_stream")
                            #     if condition2_stream is None:
                            #         condition2_stream_id = add_new_var(var_name_map, condition2_name + "_stream")
                            #         inst_stream_load = BsStreamLoad([condition2_stream_id], inst_body.condition)
                            #         inst_stream_load.stream_index = "idx"
                            #         inst.body.body.insert(inst_body_id, inst_stream_load)
                            continue
                        for operand_idx in range(inst_body.n_operand):
                            operand_id = inst_body.operands[operand_idx]
                            self.search_var_define_across_subgraph(
                                insts_stream_store_inserts,
                                insts_stream_load_inserts,
                                insts_stream_define_inserts,
                                var_define_map_subgraphs,
                                insts_subgraph_id,
                                var_name_map,
                                operand_id,
                                inst_body_id,
                            )
                    continue
                if inst.type == BsInstType.ADVANCE:
                    inst.n_operand = 1
                for operand_idx in range(inst.n_operand):
                    operand_id = inst.operands[operand_idx]
                    if operand_id in ["0", "1"] or var_name_map[operand_id] in ["0", "1"]:
                        continue
                    self.search_var_define_across_subgraph(
                        insts_stream_store_inserts,
                        insts_stream_load_inserts,
                        insts_stream_define_inserts,
                        var_define_map_subgraphs,
                        insts_subgraph_id,
                        var_name_map,
                        operand_id,
                        inst_id,
                    )

        # Insert stream store insts.
        for inst_stream_store in insts_stream_store_inserts:
            insts_subgraph_id = insts_stream_store_inserts[inst_stream_store]
            insts_subgraph = insts_subgraphs[insts_subgraph_id]
            # When the subgraph is a while loop, insert the stream store in the loop body.
            inst_while = self.get_inst_while_in_subgraph(insts_subgraph)
            if inst_while is not None:
                # Store test to test next stream
                ret_name = var_name_map[inst_stream_store.ret]
                if ret_name.startswith("test") or ret_name.startswith("accum"):
                    condition_stream_name = var_name_map[inst_stream_store.ret]
                    condition_stream_name_next = condition_stream_name + "_next"
                    condition_stream_next = add_new_var(
                        var_name_map, condition_stream_name_next
                    )
                    inst_stream_store.ret = condition_stream_next
                    var_tmp_streams_id = add_new_var(var_name_map, "tmp_streams")
                    inst_stream_define = BsStreamDefine(
                        [var_tmp_streams_id], condition_stream_next
                    )
                    insts_stream_define_inserts.append(inst_stream_define)
                    inst_while.cuda_swap_pointer.append(
                        f"swap_pointer(&{condition_stream_name}, &{condition_stream_name_next});"
                    )
                inst_while.body.body.append(inst_stream_store)
            else:
                insts_subgraph.append(inst_stream_store)

        # Insert stream load insts.
        for inst_stream_load in insts_stream_load_inserts:
            insts_subgraph_id = insts_stream_load_inserts[inst_stream_load]
            insts_subgraph = insts_subgraphs[insts_subgraph_id]

            # When the subgraph is a while loop, insert the stream load in the loop body.
            inst_while = self.get_inst_while_in_subgraph(insts_subgraph)
            if inst_while is not None:
                inst_while.body.body.insert(0, inst_stream_load)
            else:
                insts_subgraph.insert(0, inst_stream_load)

        # Insert stream define insts.
        for inst_stream_define in insts_stream_define_inserts:
            insts_subgraphs[0].insert(0, inst_stream_define)

        # Set memory boundary for stream store insts.
        for subgraph_id in range(len(insts_subgraphs)):
            inst_stream_while = self.get_inst_while_in_subgraph(
                insts_subgraphs[subgraph_id]
            )
            if inst_stream_while is not None:
                inst_stream_while.body.body = self.insert_if_inst_for_stream_store(
                    inst_stream_while.body.body
                )
            else:
                insts_subgraphs[subgraph_id] = self.insert_if_inst_for_stream_store(
                    insts_subgraphs[subgraph_id]
                )
        # sub sub graph in while loop
        for insts_subgraph in insts_subgraphs:
            inst_whole = self.get_inst_while_in_subgraph(insts_subgraph)
            if inst_whole is not None:
                insts_while_body = inst_whole.body.body
                insts_while_body_new, var_name_map_new = self.run_in_while_loop(
                    insts_while_body, var_name_map
                )
                inst_whole.body.body = insts_while_body_new

        new_insts = []
        for insts_subgraph in insts_subgraphs:
            new_insts += insts_subgraph
            new_insts.append(BsGraphBreak())

        # print ("new_insts", new_insts)
        return new_insts, var_name_map
