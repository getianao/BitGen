from itertools import chain

from .inst import BsInst, BsInstType
from .inst import get_max_advance_offset
from .inst_scan_thru_stream import BsScanThruStream
from .. import config as cfg


class BsWhile(BsInst):

    def __init__(self, operands, condition, ret=0, name: str = ""):
        ret = operands[0].body[-1].ret
        super().__init__(BsInstType.WHILE, operands, ret)
        self.condition = condition
        self.body = operands[0]
        self.cuda_swap_pointer = []

    def lower_to_bitstream(self, var_map, indent: str = ""):
        codes = []
        operand1 = self.get_var_name(var_map, self.operand1)
        codes.append(f"{indent}While (!{operand1}) {{")
        body_insts = list(chain(*self.body))
        for inst in body_insts:
            codes.append(inst.lower_to_bitstream(var_map, indent + "    "))
        codes.append(f"{indent}}}")
        code = "\n".join(codes)
        return code

    def lower_to_python(self, var_map, indent: str = ""):
        # super().lower_to_cuda(var_map, indent, var_define_map)
        codes = []
        condition = self.get_var_name(var_map, self.condition)
        codes.append(f"{indent}while ({condition}.any()):")
        codes.append(self.body.lower_to_python(var_map, indent))
        code = "\n".join(codes)
        return code

    def lower_to_torch(self, var_map, indent: str = ""):
        # super().lower_to_cuda(var_map, indent, var_define_map)
        codes = []
        condition = self.get_var_name(var_map, self.condition)
        codes.append(f"{indent}while (torch.any({condition}!=0)):")
        codes.append(self.body.lower_to_torch(var_map, indent))
        code = "\n".join(codes)
        return code

    def get_torch_module(self):
        return "BSWhile"

    def lower_to_cuda(self, var_map, indent: str = "", var_define_map: dict = None):
        # super().lower_to_cuda(var_map, indent, var_define_map)
        codes = []
        # for r in self.ret:
        #     is_new_define = self.update_var_define_map(r, var_define_map)
        #     if is_new_define:
        #         ret_name = self.get_var_name_cuda(var_map, r)
        #         codes.append(f"{indent}uint32_t " + ret_name + ";")

        graph_break_type = cfg.get_config("pass_graph_break")  # Fuse while loop

        condition_name = self.get_var_name_cuda(var_map, self.condition)

        if graph_break_type == -1:
            codes.append(f"{indent}if (threadIdx.x == 0) {{loop_counter = 0;}}")
            codes.append(f"{indent}while (!block_all_zeros({condition_name}, &zero_flag)) {{")
            codes.append(
                f"{indent}    if (threadIdx.x == 0) {{ loop_counter +=1 ; }}"
            )
        else:
            codes.append(
                f"{indent}while (!bs_all_zeros({condition_name}, n_unit_basic)) {{"
            )
        indent2 = indent + "    "
        indent3 = indent2 + "    "
        max_advance_offset = get_max_advance_offset(list(chain(self.body.body)))
        loop_start = [
            # f'{indent2}loop_count++; uint32_t ones_n = bs_ones_num({condition_name}, n_unit_basic); if (threadIdx.x == 0) printf("ones_n: %d, loop_count: %d\\n", ones_n, loop_count);',
            f"{indent2}for (uint32_t i = 0; i < ceil(1.0 * n_unit_basic / (blockDim.x - {max_advance_offset})); i += 1) {{",
            f"{indent3}int idx = i * (blockDim.x - 1) + threadIdx.x;",
        ]
        loop_end = [
            f"{indent2}}}",
            f"{indent2}__syncthreads();",
        ]

        insts = list(chain(self.body.body))
        stream_define_insts = []
        for inst_id, inst in enumerate(insts):
            if inst.type == BsInstType.STREAMDEFINE:
                stream_define_insts.append(inst)
            else:
                break
        for inst in stream_define_insts:
            codes.append(inst.lower_to_cuda(var_map, indent2, var_define_map))
        # Group insts by graph break
        insts = insts[len(stream_define_insts) :]
        subgraph_in_while = self.break_insts(insts)
        for insts in subgraph_in_while:
            insts_type = [inst.type for inst in insts]
            if BsInstType.SCANTHRU in insts_type:
                code_scanthru = self.lower_subgraph_scanthru(
                    insts, var_map, var_define_map, self.ret, indent2
                )
                codes.append(code_scanthru)
                codes.append(f"{indent2}__syncthreads();")
            if BsInstType.WHILE in insts_type:
                var_define_map_copy = var_define_map.copy()
                for inst in insts:
                    codes.append(
                        inst.lower_to_cuda(var_map, indent2, var_define_map_copy)
                    )
            elif graph_break_type == -1:
                var_define_map_copy = var_define_map.copy()
                for inst in insts:
                    codes.append(
                        inst.lower_to_cuda(var_map, indent2, var_define_map_copy)
                    )
            else:
                codes += loop_start
                var_define_map_copy = var_define_map.copy()
                for inst in insts:
                    codes.append(
                        inst.lower_to_cuda(var_map, indent3, var_define_map_copy)
                    )
                codes += loop_end
        for c in self.cuda_swap_pointer:
            codes.append(indent2 + c)
        codes.append(f"{indent}}}")
        if graph_break_type == -1:
            # codes.append(f"{indent}if (threadIdx.x == 0) {{dynamic_adv_offset = max(dynamic_adv_offset, loop_counter);}}")
            codes.append(f"{indent}if (threadIdx.x == 0) {{dynamic_adv_offset += loop_counter;}}")
        code = "\n".join(codes)
        return code

    def __str__(self):
        return f"While[{self.body}]"

    def __repr__(self):
        return self.__str__()

    def add_new_var(self, var_name_map, var_name):
        # Found existing var_id
        if var_name in var_name_map.values():
            for var_id, name in var_name_map.items():
                if name == var_name:
                    return var_id
        # Add new var_id
        new_var_id = len(var_name_map)
        var_name_map[new_var_id] = var_name
        return new_var_id

    def break_insts(self, insts):
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

    def get_var_by_name(self, var_name, var_name_map):
        for var in var_name_map:
            if var_name_map[var] == var_name:
                return var
        return None

    def lower_subgraph_scanthru(
        self, insts, var_name_map, var_define_map, output_var, indent
    ):
        inst_scanthru = None
        for insts_id, inst in enumerate(insts):
            if inst.type == BsInstType.SCANTHRU:
                inst_scanthru = inst
                break
        assert inst_scanthru is not None
        op1_name = var_name_map[inst_scanthru.operand1]
        op2_name = var_name_map[inst_scanthru.operand2]
        ret_name = var_name_map[inst_scanthru.ret]
        op1_stream_id = self.get_var_by_name(op1_name + "_stream", var_name_map)
        op2_stream_id = self.get_var_by_name(op2_name + "_stream", var_name_map)
        ret_stream_id = self.get_var_by_name(ret_name + "_stream", var_name_map)
        inst_scanthru_stream = BsScanThruStream(
            [op1_stream_id, op2_stream_id], ret_stream_id
        )
        return inst_scanthru_stream.lower_to_cuda(var_name_map, indent)
