import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

from ..inst import BsInstType, BsAssign, BsAdvance, BsIf, BsBlock, BsCall, BsStr
from ..passes.pass_utils import (
    remove_empty_insts,
    update_insts,
    build_graph,
    get_var_by_name,
)
from ..log import MyLogger
from .. import config as cfg
from ..tool import global_timer


class short_circuit_pass:
    def __init__(self, profile=False):
        self.pass_name = "short_circuit"
        self.profile = profile

    def get_and_chains(self, insts, var_name_map, G):
        and_chains = []
        roots = range(8)  # first 8 inst are basic streamload
        end_node = len(insts) - 1
        # Get all consecutive and nodes
        for root in roots:
            paths = list(nx.all_simple_paths(G, source=root, target=end_node))
            for path in paths:
                # print(path)
                and_mask = []
                # Break the path from the last CC node
                # Reverse the path

                for node in path:
                    if insts[node].type == BsInstType.AND:
                        and_mask.append(1)
                    elif insts[node].type in [
                        BsInstType.WHILE,
                        BsInstType.ASSIGN,
                    ]:
                        and_mask.append(2)  # Skip
                    else:
                        and_mask.append(0)  # Terminate

                and_chain = []
                for n, m in zip(path, and_mask):
                    if m == 1:
                        and_chain.append(n)
                    elif m == 2:
                        continue
                    elif m == 0:
                        if and_chain:
                            and_chains.append(and_chain)
                            and_chain = []
                    else:
                        raise ValueError(f"Unknown mask value: {m}")
                if and_chain:
                    and_chains.append(and_chain)

        # print("and_chains")
        # print(and_chains)
        return and_chains

    def check_and_chain_overlap(self, start_node, end_node, removed_nodes):
        if start_node >= end_node:
            raise ValueError(
                f"start_node should be less than end_node: start_node={start_node}, end_node={end_node}"
            )
        for removed_node in removed_nodes:
            started = removed_node[0]
            ended = removed_node[1]
            if start_node <= started and end_node >= started:
                return True
            if start_node >= started and start_node <= ended:
                return True
        return False

    def remove_duplicate_sub_and_chains(self, and_chains):
        # Remove duplicate chains and subchains
        and_chain_strings = [",".join(map(str, a)) for a in and_chains]
        unique_strings = []
        unique_and_chain = []
        for a_id, a_s in enumerate(and_chain_strings):
            if not any(a_s in existing for existing in unique_strings):
                unique_strings.append(a_s)
                unique_and_chain.append(and_chains[a_id])
        return unique_and_chain

    def run(self, insts, var_name_map):
        new_insts = insts.copy()
        insert_insts = {}  # inst -> pos_id
        start_time = time.time()
        # print(insts)
        G = build_graph(insts, var_name_map)
        end_time = time.time()
        MyLogger.debug(
            f"[short_circuit] build graph in {end_time - start_time:.2f} seconds"
        )
        # paths = list(nx.dfs_preorder_nodes(G, source=0))
        # print(paths)
        and_chains = self.get_and_chains(insts, var_name_map, G)

        short_circuit_start_pos = cfg.get_config("pass_short_circuit_start")
        short_circuit_interval = cfg.get_config("pass_short_circuit_interval")
        if short_circuit_interval == 0:
            short_circuit_interval = 1
        if short_circuit_interval < short_circuit_start_pos:
            short_circuit_start_pos = short_circuit_interval
        # Process and_chains.
        # and_chains start from the first advance inst.
        first_adv_inst_pos = -1
        for inst_id, inst in enumerate(insts):
            if inst.type == BsInstType.ADVANCE:
                first_adv_inst_pos = inst_id
                break
        # print("first_adv_inst_pos", first_adv_inst_pos)
        new_and_chains = []
        for and_chain in and_chains:
            for rnode_id, rnode in enumerate(and_chain):
                if rnode > first_adv_inst_pos:
                    new_and_chains.append(and_chain[rnode_id:])
                    break
        # Remove and chains with less than 3 nodes
        new_and_chains = [
            x for x in new_and_chains if len(x) > short_circuit_start_pos + 3
        ]
        # Remove duplicate and chains
        new_and_chains = self.remove_duplicate_sub_and_chains(new_and_chains)
        # new_and_chains = [list(x) for x in set(tuple(x) for x in new_and_chains)]
        # Sort by chain length
        new_and_chains = sorted(new_and_chains, key=lambda x: len(x), reverse=True)

        and_chains = new_and_chains
        # for and_chain_id, and_chain in enumerate(and_chains):
        #     MyLogger.debug(f"[short_circuit] and_chain candidate {and_chain_id}: {and_chain}")
        #     and_chain_ret_str = ""
        #     for rnode in and_chain:
        #         and_chain_ret_str += f"{var_name_map[insts[rnode].ret]}, "
        #     MyLogger.debug(
        #         f"[short_circuit] and_chain candidate {and_chain_id} name: [{and_chain_ret_str}]"
        #     )

        # Break the chain at OR nodes
        new_and_chains = []
        for and_chain in and_chains:
            break_and_chain = []
            for a_node_id in range(len(and_chain) - 1):
                a_node = and_chain[a_node_id]
                a_node_next = and_chain[a_node_id + 1]
                break_and_chain.append(a_node)
                for inst_test in insts[a_node + 1 : a_node_next]:
                    if inst_test.type == BsInstType.OR:
                        new_and_chains.append(break_and_chain)
                        break_and_chain = []
                        break
            break_and_chain.append(and_chain[-1])
            new_and_chains.append(break_and_chain)

        # Remove and chains with less than 3 nodes
        new_and_chains = [
            x for x in new_and_chains if len(x) > short_circuit_start_pos + 3
        ]
        # Remove duplicate and chains
        new_and_chains = self.remove_duplicate_sub_and_chains(new_and_chains)
        # new_and_chains = [list(x) for x in set(tuple(x) for x in new_and_chains)]
        # Sort by chain length
        new_and_chains = sorted(new_and_chains, key=lambda x: len(x), reverse=True)
        and_chains = new_and_chains

        # Avoid jump over used variables
        use_var_map = {}  # var_id -> [used_inst_id]
        for inst_id, inst in enumerate(insts):
            if inst.type in [BsInstType.WHILE, BsInstType.STR]:
                continue
            n_operands = inst.n_operand
            if inst.type == BsInstType.ADVANCE:
                n_operands = 1
            for operand in inst.operands[:n_operands]:
                if operand in use_var_map:
                    use_var_map[operand].append(inst_id)
                else:
                    use_var_map[operand] = [inst_id]
        new_and_chains = []
        for and_chain in and_chains:
            and_chain_start = and_chain[0]
            and_chain_end = and_chain[-1]
            remove_flag = False
            for and_chain_inst_id in range(and_chain_start, and_chain_end + 1):
                if insts[and_chain_inst_id].type in [BsInstType.STR]:
                    continue
                ret_id = insts[and_chain_inst_id].ret
                if not var_name_map[ret_id].startswith("at"):
                    continue
                # print(f"[short_circuit] check var {var_name_map[ret_id]}")
                if ret_id in use_var_map:
                    for use_pos in use_var_map[ret_id]:
                        # print(f"[short_circuit] check var {var_name_map[ret_id]} used at {use_pos}")
                        if use_pos > and_chain_end:
                            remove_flag = True
                            # print(f"[short_circuit] remove used var {var_name_map[ret_id]}, inst: {insts[and_chain_inst_id]}")
                            break
            if not remove_flag:
                new_and_chains.append(and_chain)
        and_chains = new_and_chains

        for and_chain_id, and_chain in enumerate(and_chains):
            MyLogger.debug(
                f"[short_circuit] and_chain candidate {and_chain_id}: {and_chain}"
            )
            and_chain_ret_str = ""
            for rnode in and_chain:
                and_chain_ret_str += f"{var_name_map[insts[rnode].ret]}, "
            MyLogger.debug(
                f"[short_circuit] and_chain candidate {and_chain_id} name: [{and_chain_ret_str}]"
            )
        removed_nodes = []
        removed_inst_num = 0
        removed_chain_num = 0
        for and_chain in and_chains:
            start_node = and_chain[short_circuit_start_pos]
            end_node = and_chain[-1]
            # Avoid deleting nodes that have been removed

            if self.check_and_chain_overlap(start_node, end_node, removed_nodes):
                continue
            else:
                removed_nodes.append([start_node, end_node])
            MyLogger.debug(
                f"[short_circuit] delete and_chain: {and_chain}. ({end_node - start_node + 1} insts, interval={short_circuit_interval})"
            )

            last_cc_adv_mem_store_inst_id = None
            for rev_and_chin_inst_id in range(end_node, start_node, -1):
                if (
                    insts[rev_and_chin_inst_id].type == BsInstType.STR
                    and "advance_memory[" in insts[rev_and_chin_inst_id].str
                ):
                    last_cc_adv_mem_store_inst_id = rev_and_chin_inst_id
                    break

            if last_cc_adv_mem_store_inst_id:
                MyLogger.debug(
                    f"[short_circuit] last_cc_adv_mem_store_inst_id: {last_cc_adv_mem_store_inst_id}"
                )
                for and_chain_node_id in reversed(and_chain):
                    if and_chain_node_id < last_cc_adv_mem_store_inst_id:
                        end_node = and_chain_node_id
                        break
            and_chain = and_chain[: and_chain.index(end_node) + 1]
            MyLogger.debug(
                f"[short_circuit] exclude cc_adv mem. and_chain: {and_chain}. ({end_node - start_node + 1} insts, interval={short_circuit_interval})"
            )

            interval_list = range(
                short_circuit_interval,
                len(and_chain[short_circuit_start_pos:]),
                short_circuit_interval,
            )
            inserted_node = []
            inserted_node2 = []
            if cfg.get_config("pass_short_circuit_syncpoint"):
                for and_chain_node_id, and_chain_node_id_next in zip(
                    interval_list, interval_list[1:]
                ):
                    # Check if there is a sync between two and nodes
                    and_chain_inst_id = and_chain[and_chain_node_id]
                    and_chain_inst_id_next = and_chain[and_chain_node_id_next]
                    and_chain_inst = insts[and_chain_inst_id]
                    ret_id = and_chain_inst.ret
                    for inst_id in range(and_chain_inst_id, and_chain_inst_id_next):
                        if (
                            insts[inst_id].type == BsInstType.STR
                            and "sync" in insts[inst_id].str
                        ):
                            MyLogger.debug(
                                f"[short_circuit] Insert block_all_zeros at sync point: inst_id={inst_id}, check={var_name_map[ret_id]}, label=LABEL{removed_chain_num}."
                            )
                            str_if_inst = BsStr(
                                [
                                    f"if (block_all_zeros({var_name_map[ret_id]}, &zero_flag)) {{ goto LABEL{removed_chain_num}; }}"
                                ]
                            )
                            str_if_inst.comment = "sync point"
                            # insert_insts[str_if_inst] = inst_id + 1
                            new_insts[inst_id] = str_if_inst
                            inserted_node.append(and_chain_inst_id)
                            break
                        elif insts[inst_id].type == BsInstType.WHILE:
                            MyLogger.debug(
                                f"[short_circuit] Insert block_all_zeros at sync point: inst_id={inst_id}, check={var_name_map[ret_id]}, label=LABEL{removed_chain_num}."
                            )
                            str_if_inst = BsStr(
                                [
                                    f"if (block_all_zeros({var_name_map[ret_id]}, &zero_flag)) {{ goto LABEL{removed_chain_num}; }}"
                                ]
                            )
                            str_if_inst.comment = "sync point"
                            insert_insts[str_if_inst] = inst_id
                            # new_insts[inst_id] = str_if_inst
                            inserted_node.append(and_chain_inst_id)
                            break

            # Insert branch with fixed interval in and_chain.
            if cfg.get_config("pass_short_circuit_interval_disable") == 0:
                for and_inst_id in interval_list:
                    and_node = and_chain[and_inst_id]  # id in insts
                    if any(
                        and_node_check in inserted_node
                        for and_node_check in and_chain[
                            and_inst_id : and_inst_id + short_circuit_interval
                        ]
                    ):
                        continue
                    and_inst = insts[and_node]
                    ret_id = and_inst.ret
                    str_if_inst = BsStr(
                        [
                            f"if (block_all_zeros({var_name_map[ret_id]}, &zero_flag)) {{ goto LABEL{removed_chain_num}; }}"
                            # f"if (block_all_zeros({var_name_map[ret_id]})) {{ if(threadIdx.x==0){{printf(\"break at {and_inst_id/short_circuit_interval}\\n\");}} goto LABEL{removed_chain_num}; }}"
                            # f"if (warp_all_zeros({var_name_map[ret_id]})) {{ if(threadIdx.x==0){{printf(\"break\\n\");}}__syncthreads();  goto LABEL{removed_chain_num};  }} __syncthreads();"
                            # f"skip_flag = warp_all_zeros({var_name_map[ret_id]});"
                        ]
                    )
                    insert_insts[str_if_inst] = and_node + 1
                    inserted_node2.append(and_node)
                    MyLogger.debug(
                        f"[short_circuit] Insert block_all_zeros at node {and_node}: ret={var_name_map[ret_id]}."
                    )

            # Define var to 0 outside the loop
            if len(inserted_node2) > 0 or len(inserted_node) > 0:
                set_zero_var_id = []
                zero_var_id = get_var_by_name("0", var_name_map)
                if last_cc_adv_mem_store_inst_id:
                    body_1_insts = insts[
                        start_node + 1 : last_cc_adv_mem_store_inst_id + 1
                    ]
                else:
                    body_1_insts = insts[start_node + 1 : end_node + 1]
                for body_1_inst in body_1_insts:
                    ret_var_id = body_1_inst.ret
                    if ret_var_id in set_zero_var_id:
                        continue
                    if body_1_inst.type == BsInstType.STR:
                        continue
                    zero_inst = BsAssign([zero_var_id], ret_var_id)
                    insert_insts[zero_inst] = start_node
                    set_zero_var_id.append(ret_var_id)

                # Insert branch label at the end of the and_chain.
                str_lable_inst = BsStr([f"LABEL{removed_chain_num}:"])
                if last_cc_adv_mem_store_inst_id:
                    while insts[last_cc_adv_mem_store_inst_id].type == BsInstType.STR:
                        last_cc_adv_mem_store_inst_id -= 1
                    insert_insts[str_lable_inst] = last_cc_adv_mem_store_inst_id + 1
                else:
                    insert_insts[str_lable_inst] = end_node + 1
                removed_chain_num += 1
                removed_inst_num += end_node - start_node + 1

            # inst_graph_break = BsCall(["torch._dynamo.graph_break"], None)
            # insert_insts[inst_graph_break] = start_node + 1

        update_insts(new_insts, insert_insts)
        new_insts = remove_empty_insts(new_insts)
        end_time = time.time()
        MyLogger.debug(
            f"[short_circuit]: process chain {end_time - start_time:.2f} seconds"
        )
        MyLogger.debug(
            f"[short_circuit] skip {removed_inst_num} insts on {removed_chain_num} chains "
        )
        if cfg.get_config("pass_inst_stats"):
            if (
                not hasattr(global_timer, "inst_result")
                or global_timer.inst_result is None
            ):
                global_timer.inst_result = {}
            global_timer.inst_result["removed_chain_num"] = (
                global_timer.inst_result.get("removed_chain_num", 0) + removed_chain_num
            )
            global_timer.inst_result["removed_inst_num"] = (
                global_timer.inst_result.get("removed_inst_num", 0) + removed_inst_num
            )

        return new_insts, var_name_map
