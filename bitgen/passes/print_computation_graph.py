import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

from ..inst import BsInstType, BsAssign, BsAdvance
from ..passes.pass_utils import build_graph


class print_computation_graph_pass:
    def __init__(self, save_path="computation_graph.pdf"):
        self.pass_name = "print_computation_graph"
        self.save_path = save_path

    def print_computation_graph(self, insts, var_name_map):
        G = build_graph(insts, var_name_map)

        node_labels = {}
        nodes = list(G.nodes)
        for inst_id in nodes:
            inst = G.nodes[inst_id]["inst"]
            # Highlight the CC output
            if isinstance(inst.ret, list):
                assert len(inst.ret) == 1
                ret_name = var_name_map[inst.ret[0]]
            else:
                ret_name = var_name_map[inst.ret]
            if ret_name.startswith("CC") and ret_name.count("adv") == 0:
                G.nodes[inst_id]["shape"] = "box"
            # Highlight the And nodes
            if inst.type == BsInstType.AND:
                G.nodes[inst_id]["fillcolor"] = "lightcoral"
            else:
                G.nodes[inst_id]["fillcolor"] = "lightblue"
            node_labels[inst_id] = insts[inst_id].operation
        # TODO(tge): Start from basic
        G.graph["node"] = {
            "shape": "circle",
            "style": "filled",
            "width": "1",
            "height": "1",
            "fontsize": "30",
            "nodesep": "1",
            "ranksep": "1",
        }
        G.graph["edge"] = {"arrowsize": "1", "fontsize": "30"}
        G.graph["graph"] = {"splines": "spline", "rankdir": "TB"}

        A = nx.nx_agraph.to_agraph(G)
        root_nodes = range(8)
        A.add_subgraph(root_nodes, rank="same")
        A.layout(prog="dot")

        # Adjust figure size based on the number of nodes
        num_nodes = len(nodes)
        fig_size = max(16, num_nodes)
        plt.figure(figsize=(fig_size, fig_size))

        A.draw(
            self.save_path,
            format="pdf",
            prog="dot",
        )

    def run(self, insts, var_name_map):
        self.print_computation_graph(insts, var_name_map)
        return insts, var_name_map
