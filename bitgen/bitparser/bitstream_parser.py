# Parse bitstream code generated from icgrep

from ..inst import *
from .. import config as cfg

import re
from enum import Enum

operation_name_map = {
    "~": BsInstType.NOT,
    "&": BsInstType.AND,
    "|": BsInstType.OR,
    "^": BsInstType.XOR,
    "pablo.Advance": BsInstType.ADVANCE,
    "pablo.MatchStar": BsInstType.MATCHSTAR,
    "pablo.ScanThru": BsInstType.SCANTHRU,
}


class expr_type(Enum):
    VARIABLE = 0
    UNARY = 1
    BINARY = 2
    FUNCTION = 3
    SELECT = 5


class BitstreamParser:

    def __init__(self):
        self.var_name_map = {}
        self.name_var_map = {}

    def add_var(self, var_name: str):
        if var_name not in self.name_var_map:
            var_id = len(self.name_var_map)
            self.name_var_map[var_name] = var_id
            self.var_name_map[var_id] = var_name
        return self.name_var_map[var_name]

    def get_var(self, var_name):
        if var_name in self.name_var_map:
            return self.name_var_map[var_name]
        else:
            raise ValueError("Invalid variable name: " + var_name)

    def parse_expr(self, expr: str):
        # Remove parenthesis
        expr = expr.strip()
        if expr[0] == "(" and expr[-1] == ")":
            expr = expr[1:-1]
            expr = expr.strip()

        # Parse fucntion call
        pattern = r"^\s*([\w\.]+)\s*\(\s*(.*?)\s*\)\s*$"
        match = re.match(pattern, expr)
        if match:
            function_name = match.group(1)
            arguments = match.group(2)
            arguments = arguments.split(",")
            arguments = [arg.strip() for arg in arguments]

            if function_name in operation_name_map:
                return expr_type.FUNCTION, [
                    operation_name_map[function_name],
                    *arguments,
                ]
            elif function_name == "Next":
                return expr_type.VARIABLE, arguments
            else:
                raise ValueError("Invalid function call:" + function_name)

        tokens = expr.split()
        if len(tokens) == 5:
            if tokens[1] == "?":
                return expr_type.SELECT, [BsInstType.SEL, tokens[0], tokens[2], tokens[4]]
            else :
                raise ValueError("Invalid expression")
        elif len(tokens) == 3:
            # binary operation
            if tokens[1] in operation_name_map:
                return expr_type.BINARY, [
                    operation_name_map[tokens[1]],
                    tokens[0],
                    tokens[2],
                ]
            else:
                raise ValueError("Invalid binary operation")
        elif len(tokens) == 2:
            # unary operation
            if tokens[0] in operation_name_map:
                return expr_type.UNARY, [operation_name_map[tokens[0]], tokens[1]]
            else:
                raise ValueError("Invalid unary operation")
        elif len(tokens) == 1:
            if tokens[0][0] == '~':
                return expr_type.UNARY, [BsInstType.NOT, tokens[0][1:]]
            return expr_type.VARIABLE, [tokens[0]]
        else:
            raise ValueError("Invalid expression: " + str(tokens))

    def get_indent(self, line: str) -> int:
        indent = len(line) - len(line.lstrip(" "))
        return indent

    def get_block_end(self, codes: list, block_start: int, indent_out: int):
        for i in range(block_start, len(codes)):
            line_next = codes[i]
            indent_next = self.get_indent(line_next)
            if indent_next == indent_out:
                return i

    def parse_branch_while(self, codes: list, line_id: int):
        line = codes[line_id].strip()
        assert line.startswith("while")
        condition = line.split("while")[1].strip().split(":")[0].strip()
        condition_id = self.add_var(condition)
        indent_out = self.get_indent(codes[line_id])
        body_1_start = line_id + 1
        body_1_end = self.get_block_end(codes, body_1_start, indent_out)
        body_1_insts = self.parse_lines(codes[body_1_start:body_1_end])
        body_1_inst = BsBlock([body_1_insts], body_1_insts[-1].ret)
        while_rets = list(set([body_1_inst.ret]))
        inst = BsWhile([body_1_inst], condition_id, while_rets)
        return body_1_end, inst

    def parse_branch_if(self, codes: list, line_id: int):
        line = codes[line_id].strip()
        assert line.startswith("if")
        condition = line.split("if")[1].strip().split(":")[0].strip()
        condition_id = self.add_var(condition)
        indent_out = self.get_indent(codes[line_id])
        body_1_start = line_id + 1
        body_1_end = self.get_block_end(codes, body_1_start, indent_out)
        body_1_insts = self.parse_lines(codes[body_1_start:body_1_end])
        body_1_inst = BsBlock([body_1_insts], body_1_insts[-1].ret)
        body_2_inst = None
        body_2_end = body_1_end
        if codes[body_1_end].strip().startswith("else"):
            # Parse else block
            body_2_start = body_1_end + 1
            body_2_end = self.get_block_end(codes, body_2_start, indent_out)
            body_2_insts = self.parse_lines(codes[body_2_start:body_2_end])
            body_2_inst = BsBlock([body_2_insts], body_1_insts[-1].ret)
        if_rets = list(set([body_1_inst.ret, body_2_inst.ret]))  # unique ret list
        inst = BsIf([body_1_inst, body_2_inst], condition_id, if_rets)
        return body_2_end, inst

    def parse_line(self, codes: list, line_id: int):
        # print(f"Parse line {line_id}:", codes[line_id])

        line = codes[line_id].strip()
        index = line.find("#")
        if index != -1:
            line = line[:index].strip()
        if line == "":
            return line_id + 1, None

        # Parse branch
        if line.startswith("while"):
            next_line_id, inst = self.parse_branch_while(codes, line_id)
            return next_line_id, inst
        if line.startswith("if"):
            next_line_id, inst = self.parse_branch_if(codes, line_id)
            return next_line_id, inst

        # Parse statement
        expr_left, expr_right = line.split("=")
        type_left, arguments_left = self.parse_expr(expr_left)
        type_right, arguments_right = self.parse_expr(expr_right)
        assert type_left == expr_type.VARIABLE
        # print("arguments_left:", arguments_left)

        var_left = self.add_var(arguments_left[0])

        if type_right == expr_type.VARIABLE:
            var_right = self.get_var(arguments_right[0])
            inst = BsAssign([var_right], var_left)
        elif type_right == expr_type.UNARY:
            if arguments_right[0] == BsInstType.NOT:
                var_right = self.get_var(arguments_right[1])
                inst = BsNot([var_right], var_left)
            else:
                raise ValueError("Invalid unary operation")
        elif type_right == expr_type.BINARY:
            var_right_1 = self.get_var(arguments_right[1])
            var_right_2 = self.get_var(arguments_right[2])
            if arguments_right[0] == BsInstType.AND:
                inst = BsAnd([var_right_1, var_right_2], var_left)
            elif arguments_right[0] == BsInstType.OR:
                inst = BsOr([var_right_1, var_right_2], var_left)
            elif arguments_right[0] == BsInstType.XOR:
                inst = BsXor([var_right_1, var_right_2], var_left)
            else:
                raise ValueError("Invalid binary operation")
        elif type_right == expr_type.FUNCTION:

            if arguments_right[0] == BsInstType.ADVANCE:
                var_right_1 = self.get_var(arguments_right[1])
                var_right_2 = arguments_right[2]
                inst = BsAdvance([var_right_1, var_right_2], var_left)
            elif arguments_right[0] == BsInstType.MATCHSTAR:
                var_right_1 = self.get_var(arguments_right[1])
                var_right_2 = self.get_var(arguments_right[2])
                inst = BsMatchStar([var_right_1, var_right_2], var_left)
                raise NotImplementedError("MatchStar is not implemented")
            elif arguments_right[0] == BsInstType.SCANTHRU:
                var_right_1 = self.get_var(arguments_right[1])
                var_right_2 = self.get_var(arguments_right[2])
                inst = BsScanThru([var_right_1, var_right_2], var_left)
                raise NotImplementedError("ScanThru is not implemented")
            else:
                raise ValueError("Invalid function call")
        elif type_right == expr_type.SELECT:
            condition = self.get_var(arguments_right[1])
            var_right_2 = self.get_var(arguments_right[2])
            var_right_3 = self.get_var(arguments_right[3])
            if cfg.get_config("backend") == "cuda":
                # Separate it into three operations
                not_condition = self.add_var("sel_not_" + arguments_right[1])
                mask_var_right_2 = self.add_var("sel_mask_" + arguments_right[2])
                mask_var_right_3 = self.add_var("sel_mask_" + arguments_right[3])
                inst_not = BsNot([condition], not_condition)
                inst_mask_2 = BsAnd([condition, var_right_2], mask_var_right_2)
                inst_mask_3 = BsAnd([not_condition, var_right_3], mask_var_right_3)
                inst_mask_other = BsOr([mask_var_right_2, mask_var_right_3], var_left)
                return line_id + 1, [inst_not, inst_mask_2, inst_mask_3, inst_mask_other]
            else:
                inst = BsSel([var_right_2, var_right_3], condition, var_left)
        else:
            raise ValueError("Invalid expression")
        return line_id + 1, inst

    def parse_lines(self, codes: list) -> list:
        insts = []
        line_id = 0
        while line_id < len(codes):
            next_line_id, inst = self.parse_line(codes, line_id)
            line_id = next_line_id
            if inst is not None:
                if isinstance(inst, list):
                    insts.extend(inst)
                elif isinstance(inst, BsInst):
                    insts.append(inst)
                else:
                    raise ValueError("Invalid instruction")
        return insts

    def parse_file(self, file_path) -> list:
        insts = []
        if cfg.get_config("backend") == "cuda":
            basis_stream_var = self.add_var("basic_stream")
            for i in range(8):
                basis_var = self.add_var(f"basis{i}")
                inst = BsStreamLoad([basis_stream_var], basis_var)
                inst.stream_index = f"idx + {i} * n_unit_basic"
                insts.append(inst)
        elif cfg.get_config("backend") == "torch":
            for i in range(8):
                basis_var = self.add_var(f"basis{i}")
                inst = BsAssign([f"basic_stream[{i}]"], basis_var)
                insts.append(inst)

        self.add_var("0")
        self.add_var("1")
        with open(file_path, "r") as f:
            codes = f.read().split("\n")
            insts.extend(self.parse_lines(codes))

        if cfg.get_config("backend") == "cuda":
            var_result_stream = self.add_var(f"bs_result_stream")
            inst_stream_store = BsStreamStore([insts[-1].ret], var_result_stream)
            inst_stream_store.use_or = True
            # inst_stream_store.use_atomic_or = True
            inst_stream_store.stream_index = "idx"
            insts.append(inst_stream_store)

        # print("insts:", insts)
        return insts
