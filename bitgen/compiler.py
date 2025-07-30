from itertools import chain

from .nfa import NFA, State
from .bitstream import Bitstream
from .inst import *
from .compiler_helper import *
from .codegen import Generator, PythonGenerator, CudaGenerator


def print_list(l, end="\n"):
    print("[", end="")
    for i in l:
        if isinstance(i, list):
            print_list(i, end="")
        else:
            print(str(i) + ", ", end="")
    print("]", end=end)

def in_list(l, item):
    for i in l:
        if isinstance(i, list):
            if in_list(i, item):
                return True
        elif i == item:
            return True
    return False

class BitstreamCompiler:
    def __init__(self, nfa: NFA):
        self.nfa = nfa
        self.insts = []
        self.paths_state = []
        self.paths_inst = []
        self.var_name_map = {}  # var_id -> var_name
        self.name_var_map = {}  # var_name -> var_id
        self.var_prefix_count = {}

        self.cc_stream = {}  # state_id -> cc_var_id
        self.cc_insts = []

    def get_var_name_by_id(self, var_id: int):
        return self.var_name_map[var_id]

    def get_var_id_by_name(self, var_name: str):
        return self.name_var_map[var_name]

    def add_var(self, var_name: str):
        if var_name not in self.name_var_map:
            self.name_var_map[var_name] = len(self.name_var_map)
            self.var_name_map[self.name_var_map[var_name]] = var_name
        return self.name_var_map[var_name]

    def create_var(self, var_name_prefeix: str):
        if var_name_prefeix not in self.var_prefix_count:
            self.var_prefix_count[var_name_prefeix] = 0
        var_name = f"{var_name_prefeix}_{self.var_prefix_count[var_name_prefeix]}"
        self.var_prefix_count[var_name_prefeix] += 1
        return self.add_var(var_name)

    def check_duplicate_list(self, state_id_paths):
        seen = set()
        for state_id_path in state_id_paths:
            t = tuple(state_id_path)
            if t in seen:
                assert "Duplicate found: %s" % str(state_id_path)
            else:
                seen.add(t)

    def lower(self, type: str, file_path: str):
        # self.check_duplicate_list(self.paths_state)
        with open(file_path, "w") as f:
            if type == "bitstream":
                Gen = Generator
            elif type == "python":
                Gen = PythonGenerator
            elif type == "cuda":
                Gen = CudaGenerator
            else:
                raise ValueError("Unknown lower type.")
            gen = Gen(
                f,
                self.insts,
                self.cc_insts,
                self.paths_inst,
                self.paths_state,
                self.var_name_map,
            )
            gen.lower()

    def find_state_in_path(self, state_id, path, level=0):
        level_next = level
        if isinstance(path, list):
            for item in path:
                found, level_item = self.find_state_in_path(
                    state_id, item, level=level + 1
                )
                level_next = max(level_item, level_next)
                if found:
                    return True, level_next
            return False, level
        elif isinstance(path, int):
            if path == state_id:
                return True, level_next
            else:
                return False, level_next

    def tranform_loop(self, insts, state_path_to_state, neighbor_id):
        print(insts)
        has_loop, level = self.find_state_in_path(neighbor_id, state_path_to_state)
        if has_loop:
            print("level:", level)
            assert level == 1

    # @return insts: insts AFTER the state, included
    # @return state_path_from_state: state id in path AFTER the state, included
    # @return block_res_var
    def compile_state(
        self,
        state: State,
        marker_stream: Bitstream,
        state_path_to_state: list,  # Excluded state,
        insts_before, 
    ):
        print("~~~~~ state", state.id)
        print("~~~~~ state_path_to_state_excluded", state_path_to_state)
        insts = []

        state_insts = []
        # Check cc strream already exists
        if self.cc_stream.get(state.id) is None:
            cc_var = self.create_var("cc")
            inst_match = BsMatch(
                [self.input_stream_var, f"{state.symbol_set_expr}"],
                cc_var,
            )
            self.cc_insts.append(inst_match)
            self.cc_stream[state.id] = cc_var
        else:
            cc_var = self.cc_stream[state.id]

        inst_and = BsAnd(
            [marker_stream, cc_var],
            self.create_var("and"),
        )
        state_insts.append(inst_and)
        inst_advance = BsAdvance(
            [inst_and.ret, 1],
            self.create_var("adv"),
        )
        state_insts.append(inst_advance)
        if state.is_report():
            inst_or = BsOr(
                [self.out_stream_var, inst_advance.ret],
                self.out_stream_var,
            )
            inst_or.ret_type = ""
            state_insts.append(inst_or)

        inst_state = BsState([state_insts], inst_advance.ret, state.id)
        insts.append(inst_state)

        if state.neighbors is None or len(state.neighbors) == 0:
            print("no neighbor:", insts)
            return insts, [state.id], inst_state.ret

        block_body_insts = []
        block_body_state_paths = []
        block_body_result_marker_streams = []
        state_path_from_state = []  # include the state
        result_marker_streams = None
        # One neignbor is stored in Block, and multiple neighbors's results are connected by or
        neighbor_not_in_loop = []
        for neighbor in state.neighbors:
            found, level = self.find_state_in_path(neighbor.id, state_path_to_state)
            if found:
                print(f"Loop found. {neighbor.id} {state_path_to_state + [state.id]}")
                self.tranform_loop(
                    insts_before + insts, state_path_to_state + [state.id], neighbor.id
                )
                continue
            # # if neighbor.id in state_path:
            # if in_list(state_path, neighbor.id):
            #     print_list(state_path + [neighbor.id])
            #     print("Loop found.")
            #     continue

            block_body_inst, block_state_path, result_marker_stream = (
                self.compile_state(
                    neighbor,
                    inst_advance.ret,
                    state_path_to_state + [state.id],
                    insts_before + insts,
                )
            )
            block_body_insts.append(block_body_inst)
            block_body_state_paths.append(block_state_path)
            block_body_result_marker_streams.append(result_marker_stream)
            neighbor_not_in_loop.append(neighbor)

        if len(block_body_insts) == 1:
            insts += block_body_insts[0]
            block_res_var = block_body_result_marker_streams[0]
            state_path_from_state = [state.id] + block_body_state_paths[0]
        else:
            block_result_vars = []
            block_insts = []
            for idx, block_body_inst in enumerate(block_body_insts):
                print(f"body_{idx}: {block_body_inst}")
                block_inst = BsBlock(
                    [block_body_inst], block_body_result_marker_streams[idx],
                    name = f"From state {state.id} to {state.neighbors[idx].id}"
                )
                block_result_vars.append(block_inst.ret)
                block_insts.append(block_inst)
                state_path_from_state.append([state.id] + block_body_state_paths[idx])

            # The results of multiple neighbors are connect by OR
            block_res_var = self.create_var("block_res")
            for idx, block_result_var in enumerate(block_result_vars):
                if idx == 0:
                    assign_inst = BsAssign([block_result_var], block_res_var)
                    block_insts.append(assign_inst)
                else:
                    or_inst = BsOr([block_result_var, block_res_var], block_res_var)
                    or_inst.ret_type = ""
                    block_insts.append(or_inst)
            insts += block_insts

        # state_path.append(block_state_paths)

        # insts only store the inst starting from this state
        # insts = [[state match], [block1, block 2, final]]
        print("inst", insts)
        print("state_path_from_state", state_path_from_state)
        return insts, state_path_from_state, block_res_var

    def compile(self):

        self.input_stream_var = self.add_var("input_stream")
        self.out_stream_var = self.add_var("bs_result")
        always_active_states = self.nfa.get_always_active_states()
        start_states = self.nfa.get_start_states()

        states = start_states + always_active_states
        for idx, state in enumerate(states):
            if state.start == 0:
                inst_assign = BsCall(
                    ["create_start", f"len(input_stream) + 1"],
                    self.create_var("start"),
                )
            elif state.start == 1:
                inst_assign = BsCall(
                    ["create_ones", f"len(input_stream) + 1"],
                    self.create_var("start"),
                )
            marker_stream = inst_assign.ret

            insts, path_from_state_included, result_marker_stream = self.compile_state(
                state, marker_stream, [], []
            )
            path = path_from_state_included
            print("path:")
            print_list(path)
            print(insts)

        print("Compilation done.")

        self.paths_inst.append(insts)
        self.paths_state.append(path)
