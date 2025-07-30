import re
import numpy as np
import textwrap
import inspect
import math
import time
import os
import shutil
import sys

import torch
import torch.nn as nn
import torch.profiler
from torchviz import make_dot

from pytorch_memlab import profile as mem_profile

import triton
import triton.language as tl

from bitgen.tool import global_timer
from bitgen.bitstream import Bitstream
from bitgen import config as cfg
from bitgen.cuda import cub_exclusive_scan

# bs_debug = True
# bs_debug = False

def test_print():
    print("test")

def safe_copy(file_path, out_dir, dst=None):
    is_dir = os.path.isdir(file_path)
    copy = shutil.copytree if is_dir else shutil.copy
    name = dst or os.path.basename(file_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(os.path.join(out_dir, name)):
        copy(file_path, os.path.join(out_dir, name))
    else:
        base, extension = os.path.splitext(name)
        i = 1
        while os.path.exists(
            os.path.join(out_dir, "{}_{}{}".format(base, i, extension))
        ):
            i += 1
        copy(file_path, os.path.join(out_dir, "{}_{}{}".format(base, i, extension)))


def load_input_stream(input_path):
   with open(input_path, "rb") as f:
        input_stream = f.read()
        return input_stream

def print_callee_line_number():
    caller_frame = inspect.currentframe().f_back.f_back
    line_number = caller_frame.f_lineno
    return line_number


def print_np_as_bs(data):
    binary_string = "".join(str(int(b)) for b in data[:64])
    wrapped = textwrap.wrap(binary_string, 8)
    formatted_binary = " ".join(wrapped)
    aligned = f"{formatted_binary: <10}"
    return print(f"{aligned}")


def print_ts_as_bs(data: torch.Tensor, start_bit_id=0, max_bit_len=64, msg=None):
    assert isinstance(data, torch.Tensor)
    assert data.dim() == 1
    # assert data.dtype == torch.uint8
    datasize = data.element_size() * 8

    start_byte_id = math.floor(start_bit_id / 8)
    max_len = min(int(max_bit_len / 8), len(data) - start_byte_id)

    binary_string = "".join(
        format(b.item(), f"0{datasize}b") for b in data[start_byte_id : start_byte_id + max_len]
    )
    wrapped = textwrap.wrap(binary_string, 8)
    formatted_binary = " ".join(wrapped)
    aligned = f"{formatted_binary: <10}"  # Left-aligned and fill to 10 characters
    if msg is not None:
        print(f"{msg:<32}{aligned}")
    else:
        print(f"{aligned}")


def bs_match(input_stream: str, symbol_set_expr: str):
    result = torch.zeros(len(input_stream) + 1, dtype=torch.uint8)
    for i, symbol in enumerate(input_stream):
        result[i] = torch.uint8(re.fullmatch(symbol_set_expr, symbol))
    return result


class BSNot(nn.Module):
    def __init__(self):
        super(BSNot, self).__init__()

    def forward(self, a):
        r = torch.bitwise_not(a)
        # r = (-1 - int(a))
        return r


class BSAnd(nn.Module):
    def __init__(self):
        super(BSAnd, self).__init__()

    def forward(self, a, b):
        r = torch.bitwise_and(a, b)
        return r


class BSOr(nn.Module):
    def __init__(self):
        super(BSOr, self).__init__()

    def forward(self, a, b):
        r = torch.bitwise_or(a, b)
        return r


class BSXor(nn.Module):
    def __init__(self):
        super(BSXor, self).__init__()

    def forward(self, a, b):
        r = torch.bitwise_xor(a, b)
        return r


class BSSel(nn.Module):
    def __init__(self):
        super(BSSel, self).__init__()

    def forward(self, a, b):
        raise NotImplementedError
        r = a
        return r

@triton.jit
def advance(X, n_elements, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_M
    range_m = tl.arange(0, BLOCK_M)
    x = tl.load(X + offset + range_m)
    x = x + 1
    tl.store(X + offset + range_m, x)


class BSAdvance(nn.Module):
    def __init__(self):
        super(BSAdvance, self).__init__()

    def forward(self, a, n: int):
        assert a.dtype == torch.uint8
        assert n > 0
        assert len(a) > 0
        # return a

        # n_elements = a.numel()
        # block_size = 1024
        # grid = (triton.cdiv(n_elements, block_size),)
        # advance[grid](
        #     a,
        #     n_elements,
        #     block_size,
        #     num_warps=4,  # Adjust based on your GPU
        # )
        # return a + 1
        if n > 7:
            n = n % 8
            n_roll = math.ceil(n / 8)
            a[:n_roll] = 0
            a = torch.roll(a, shifts=n_roll, dims=0)
        if n == 0:
            return a

        # Extract the carry bits by shifting left (8 - n)
        carry = a << (8 - n)  # & 0xFF  # Extract the top n bits to carry over
        # Shift the original tensor right by n bits
        r = a >> n  # & 0xFF  # Ensure we stay within 8 bits
        # To handle carry-over, shift the carry bits to the next byte
        carry_shifted = torch.roll(carry, shifts=1, dims=0)
        carry_shifted[0] = 0  # No carry for the last byte
        r |= carry_shifted
        return r


def BSAdvanceRightFunction(a, n: int):
    assert a.dtype == torch.uint8
    assert n > 0
    assert len(a) > 0
    if n > 7:
        n = n % 8
        n_roll = math.ceil(n / 8)
        a[:n_roll] = 0
        a = torch.roll(a, shifts=n_roll, dims=0)
    if n == 0:
        return a

    # Extract the carry bits by shifting left (8 - n)
    carry = a << (8 - n)  # & 0xFF  # Extract the top n bits to carry over
    # Shift the original tensor right by n bits
    r = a >> n  # & 0xFF  # Ensure we stay within 8 bits
    # To handle carry-over, shift the carry bits to the next byte
    carry_shifted = torch.roll(carry, shifts=1, dims=0)
    carry_shifted[0] = 0  # No carry for the last byte
    r |= carry_shifted
    return r

def BSAdvanceLeftFunction(a, n: int):
    assert a.dtype == torch.uint8
    assert n > 0
    assert len(a) > 0

    if n > 7:
        n = n % 8
        n_roll = math.ceil(n / 8)
        a[:n_roll] = 0
        a = torch.roll(a, shifts=0 - n_roll, dims=0)

    if n == 0:
        return a

    # n = n % 8
    # Extract the carry bits by shifting left (8 - n)
    carry = a >> (8 - n)  # & 0xFF  # Extract the top n bits to carry over
    # Shift the original tensor right by n bits
    r = a << n  # & 0xFF  # Ensure we stay within 8 bits
    # To handle carry-over, shift the carry bits to the next byte
    carry[0] = 0x00
    carry_shifted = torch.roll(carry, shifts=-1, dims=0)
    # carry_shifted[-1] = 0  # No carry for the last byte
    r |= carry_shifted
    return r


def BSRollLeftFunction(a, n: int):
    assert a.dtype == torch.uint8
    assert n > 0
    assert len(a) > 0

    if n > 7:
        n = n % 8
        n_roll = math.ceil(n / 8)
        # a[:n_roll] = 0
        a = torch.roll(a, shifts=0 - n_roll, dims=0)

    if n == 0:
        return a

    # n = n % 8
    # Extract the carry bits by shifting left (8 - n)
    carry = a >> (8 - n)  # & 0xFF  # Extract the top n bits to carry over
    # Shift the original tensor right by n bits
    r = a << n  # & 0xFF  # Ensure we stay within 8 bits
    # To handle carry-over, shift the carry bits to the next byte
    # carry[0] = 0x00
    carry_shifted = torch.roll(carry, shifts=-1, dims=0)
    # carry_shifted[-1] = 0  # No carry for the last byte
    r |= carry_shifted
    return r

def BSRollRightFunction(a, n: int):
    assert a.dtype == torch.uint8
    assert n > 0
    assert len(a) > 0

    if n > 7:
        n = n % 8
        n_roll = math.ceil(n / 8)
        # a[:n_roll] = 0
        a = torch.roll(a, shifts=n_roll, dims=0)

    if n == 0:
        return a

    # n = n % 8
    # Extract the carry bits by shifting left (8 - n)
    carry = a << (8 - n)  # & 0xFF  # Extract the top n bits to carry over
    # Shift the original tensor right by n bits
    r = a >> n  # & 0xFF  # Ensure we stay within 8 bits
    # To handle carry-over, shift the carry bits to the next byte
    # carry[0] = 0x00
    carry_shifted = torch.roll(carry, shifts=1, dims=0)
    # carry_shifted[-1] = 0  # No carry for the last byte
    r |= carry_shifted
    return r

# @triton.jit
# def overflow_op(a, b):
#     # (a->carry & b->max) | (b->carry)
#     return ((a & (b >> 1)) | b) & 1


# @triton.jit
# def carry_add(X, Z, n_elements, BLOCK_M: tl.constexpr):
#     pid = tl.program_id(0)
#     block_start = pid * BLOCK_M
#     offset = block_start + tl.arange(0, BLOCK_M)
#     mask = offset < n_elements
#     x = tl.load(X + offset , mask=mask, other=0).to(tl.int32)
#     z = tl.associative_scan(x, 0, overflow_op)
#     z = z.to(tl.uint8)
#     tl.store(Z + offset + range_m, z, mask=mask)

@torch.compile
def brev(x):
    bits = 8
    mask = 2 ** torch.arange(bits, device=x.device, dtype=x.dtype)
    mask_rev = mask.flip(0).byte()
    b = x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
    b = b.matmul(mask_rev)
    return b


class BSAdd(nn.Module):
    def __init__(self, block_size=1024):
        super(BSAdd, self).__init__()
        self.block_size = block_size

    def forward(self, a, b):

        assert (
            a.dtype == torch.uint8 and b.dtype == torch.uint8
        ), "Inputs must be uint8 tensors"
        assert a.shape == b.shape, "Input tensors must have the same shape"
        a = brev(a)
        b = brev(b)
        c = torch.add(a, b)
        carry = c < a
        max = c == 0xFF
        max_carry = max * 2 + carry
        max_carry = max_carry.to(torch.uint8)
        cub_exclusive_scan(max_carry)
        result = max_carry + c
        result = result.to(torch.uint8)
        result = brev(result)
        return result


class BSAddOriginal(nn.Module):
    def __init__(self):
        super(BSAdd, self).__init__()

    def forward(self, a, b):
        assert len(a) == len(b)
        # tensor_c = torch.add(a, b)
        # carry = tensor_c < a
        # max = tensor_c == 0xFF

        # size = carry.size(0)
        # result_inclusive_scan = torch.zeros(size, dtype=torch.uint8, device=a.device)
        # for i in range(size):
        #     if i == 0:
        #         result_inclusive_scan[i] = 0 | carry[i]
        #     else:
        #         result_inclusive_scan[i] = (result_inclusive_scan[i - 1] & max[i]) | carry[i]
        # result_exclusive_scan = result_inclusive_scan.roll(1)
        # result_exclusive_scan[0] = 0
        # # print("result_exclusive_scan: ", result_exclusive_scan)
        # result = result_exclusive_scan + tensor_c
        # return result

        size = a.size(0)
        result = torch.zeros(size, dtype=torch.uint8, device=a.device)
        carry = 0
        for i in range(size):
            result[i] = a[i] + b[i] + carry
            carry = result[i] < a[i]
        return result


class BSScanThru(nn.Module):
    def __init__(self):
        super(BSScanThru, self).__init__()
        self.bs_add = BSAdd()
        self.bs_not = BSNot()
        self.bs_and = BSAnd()

    def forward(self, a, b):
        # ScanThru(M, C) = (C + M) ∧ ~C
        m_plus_c = self.bs_add(a, b)
        not_c = self.bs_not(b)
        c_plus_m_and_not_c = self.bs_and(m_plus_c, not_c)
        return c_plus_m_and_not_c


class BSMatchStar(nn.Module):
    def __init__(self):
        super(BSMatchStar, self).__init__()
        self.bs_and = BSAnd()
        self.bs_add = BSAdd()
        self.bs_xor = BSXor()
        self.bs_or = BSOr()

    # TODO(tge): onnx failed
    def forward(self, a, b):
        # MatchStar(M, C) = (((M ∧ C) + C) ⊕ C) ∨ M
        m_and_c = self.bs_and(a, b)
        m_and_c_plus_c = self.bs_add(m_and_c, b)
        m_and_c_plus_c_xor_c = self.bs_xor(m_and_c_plus_c, b)
        m_and_c_plus_c_xor_c_or_m = self.bs_or(m_and_c_plus_c_xor_c, a)
        return m_and_c_plus_c_xor_c_or_m


class BSIsZero(nn.Module):
    def __init__(self):
        super(BSIsZero, self).__init__()

    def forward(self, a):
        return not torch.any(a)


class CreateZeros(nn.Module):
    def __init__(self):
        super(CreateZeros, self).__init__()

    def forward(self, n):
        return torch.zeros(n, dtype=torch.torch.uint8)


class CreateOnes(nn.Module):
    def __init__(self):
        super(CreateOnes, self).__init__()

    def forward(self, n):
        return torch.ones(n, dtype=torch.torch.uint8)


class CreateStart(nn.Module):
    def __init__(self):
        super(CreateStart, self).__init__()

    def forward(self, n):
        r = torch.zeros(n, dtype=torch.uint8)
        r[0] = True
        return r


def bs_is_zero(a):
    return not torch.any(a)


def create_zeros(n: int, device=torch.device("cuda")):
    n = math.ceil(n / 8)
    return torch.zeros(n, dtype=torch.uint8, device=device)


def create_ones(n: int, device=torch.device("cuda")):
    n = math.ceil(n / 8)
    return torch.ones(n, dtype=torch.uint8, device=device)


def create_start(n: int, device=torch.device("cuda")):
    n = math.ceil(n / 8)
    r = torch.zeros(n, dtype=torch.uint8, device=device)
    r[0] = True
    return r


# def swap_endian(value):
#     value &= 0xFFFFFFFF
#     # Swap the bytes
#     swapped = (
#         ((value & 0x000000FF) << 24)
#         | ((value & 0x0000FF00) << 8)
#         | ((value & 0x00FF0000) >> 8)
#         | ((value & 0xFF000000) >> 24)
#     )
#     return swapped




def export_onnx(basic_stream_tesnor, n_char):
    model = KernelGenerated(basic_stream_tesnor.size(1))
    model.eval()
    model = torch.jit.script(model)
    onnx_path = "code_generated.onnx"
    torch.onnx.export(
        model=model,
        args=(basic_stream_tesnor,),
        input_names=["basic_stream"],
        output_names=["bs_result"],
        f=onnx_path,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        opset_version=20,
        autograd_inlining=False,
    )
    print(f"Exported onnx model to: {onnx_path}")


layer_counter = 0


def print_layer_tensor(model, start_bit_id = 0, enable=True):
    # Hook function
    def print_tensor_hook(name):
        def hook(module, input, output):
            global layer_counter
            try:
                # Get the calling frame
                frame = inspect.currentframe()
                while frame:
                    frame = frame.f_back
                    # Look for a frame from the current module
                    if frame.f_code.co_name == "forward":
                        source_file = frame.f_code.co_filename
                        line_number = frame.f_lineno
                        break
                else:
                    source_file, line_number = "Unknown", "Unknown"
                # Print layer information

                print(f"{layer_counter} Layer Name: {name}")
                print(f"Defined in: {source_file}:{line_number}")
            except Exception:
                print(f"{layer_counter} Layer Name: {name}")
                print("Source file and line number not available.")
            # Print input tensors
            layer_counter += 1
            for i, ts in enumerate(input):
                if isinstance(ts, torch.Tensor):
                    print_ts_as_bs(
                        ts, start_bit_id=start_bit_id, msg=f"Input[{i}] {ts.shape}:"
                    )
                else:
                    print(f"Input[{i}]: {ts}")
            # Print output tensor
            print_ts_as_bs(
                output, start_bit_id=start_bit_id, msg=f"Output: {output.shape}"
            )
            print("-" * 50)

        return hook

    if enable:
        # Register hooks for computation layers only
        for name, layer in model.named_modules():
            if len(list(layer.children())) == 0:
                layer.register_forward_hook(print_tensor_hook(name))


def print_layer_zeros(model, enable=True):
    # Hook function
    def print_tensor_zeros(name):
        def hook(module, input, output):
            try:
                # Get the calling frame
                frame = inspect.currentframe()
                while frame:
                    frame = frame.f_back
                    # Look for a frame from the current module
                    if frame.f_code.co_name == "forward":
                        source_file = frame.f_code.co_filename
                        line_number = frame.f_lineno
                        break
                else:
                    source_file, line_number = "Unknown", "Unknown"
                # Print layer information

                print(f"Layer Name: {name}")
                print(f"Defined in: {source_file}:{line_number}")
            except Exception:
                print(f"Layer Name: {name}")
                print("Source file and line number not available.")
            # Print input tensors
            for i, ts in enumerate(input):
                if isinstance(ts, torch.Tensor):
                    nonzeros = torch.count_nonzero(ts)
                    percent = nonzeros / ts.numel() * 100
                    print(f"input[{i}] shape: {ts.shape}, nonzeros: {percent:.2f} %")

            nonzeros = torch.count_nonzero(output)
            percent = nonzeros / output.numel() * 100
            print(f"output shape: {output.shape}, nonzeros: {percent:.2f} %, count: {nonzeros}")

        return hook

    if enable:
        # Register hooks for computation layers only
        for name, layer in model.named_modules():
            if len(list(layer.children())) == 0:
                layer.register_forward_hook(print_tensor_zeros(name))


def init_kernel(regex_func_num, input_stream_tensor, basic_stream_tensor=None, cfg={}):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_debug_torch = cfg.get("debug_torch", False)
    cfg_print_bitstream = cfg.get("print_bitstream", False)
    cfg_print_bit_start = cfg.get("print_bit_start", 0)
    cfg_profile = cfg.get("profile", False)

    cfg_warmup_iters = cfg.get("warmup_iters", 3)
    cfg_exec_iters = cfg.get("exec_iters", 3)
    cfg_torch_compile = cfg.get("torch_compile", False)
    cfg_print_zeros = cfg.get("print_zeros", False)

    # Model
    n_char = input_stream_tensor.size(0)
    model = KernelGenerated(n_char)

    torch._dynamo.reset
    if cfg_debug_torch:
        torch._inductor.config.force_disable_caches = True
        torch._dynamo.config.debug_dir_root = os.path.join(
            os.environ["BITGEN_ROOT"], "debug"
        )
        safe_copy(os.path.abspath(__file__), torch._dynamo.utils.get_debug_dir())

    options = {
        "max_autotune": True,
        "coordinate_descent_tuning": False,
        "triton.cudagraphs": False,
        "epilogue_fusion": False,
        "trace.enabled": bool(cfg_debug_torch),
        "trace.graph_diagram": bool(cfg_debug_torch),
    }

    if cfg_torch_compile:
        model = torch.compile(
            model,
            # fullgraph=True,
            dynamic=False,
            backend="inductor",
            # mode="max-autotune", # mode="reduce-overhead"
            options=options,
        )

    model = model.to(device)
    model.eval()

    # Input
    if basic_stream_tensor is None:
        assert NotImplementedError
        basic_stream_tensor = transpose_byte_to_bitstream(input_stream_tensor)
    else:
        assert isinstance(basic_stream_tensor, torch.Tensor)

    if cfg_print_bitstream:
        print_ts_as_bs(input_stream_tensor, msg="input:")
        for i in range(8):
            print_ts_as_bs(basic_stream_tensor[i], msg=f"basis[{i}]:")

    def run_torch(basic_stream_tensor):
        split_num = basic_stream_tensor.size(0)
        multi_num = basic_stream_tensor.size(1)
        assert split_num == 1
        results_list = []
        for i in range(split_num):
            for j in range(multi_num):
                # Process multi input serially
                result = model(basic_stream_tensor[i, j])
                results_list.append(result)
        results = torch.cat(results_list, dim=0)
        return results

    assert basic_stream_tensor.dim() == 4
    return None, None, None, run_torch

# def exec_kernel(
#     input_stream_tensor,
#     basic_stream_tensor=None,
#     cuda_stream=None,
#     cfg={}
# ):


#     # # Warmup
#     # if device.type == "cuda":
#     #     torch.cuda.synchronize()
#     # with torch.no_grad():
#     #     for _ in range(cfg_warmup_iters):

#     #         _ = run_func(basic_stream_tensor)
#     #         if device.type == "cuda":
#     #             torch.cuda.synchronize()

#     # Run
#     with torch.no_grad():
#         for _ in range(cfg_exec_iters):
#             with global_timer.time("run_regex"):
#                 result = run_func(basic_stream_tensor)
#     return


#     # Profile
#     if cfg_profile:
#         with torch.profiler.profile(
#             activities=[
#                 torch.profiler.ProfilerActivity.CPU,
#                 torch.profiler.ProfilerActivity.CUDA,
#             ],
#             on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler'),
#             record_shapes=True,
#             with_stack=True,
#         ) as prof:
#             with torch.no_grad():
#                 _ = run_func(basic_stream_tensor)
#         print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#         # prof.export_chrome_trace("trace.json")

#     # # export_onnx(basic_stream_tensor, n_char)
#     # print_layer_tensor(model, start_bit_id=cfg_print_bit_start, enable=cfg_print_bitstream)
#     # print_layer_zeros(model, enable=cfg_print_zeros)

#     # # explanation = torch._dynamo.explain(model, basic_stream_tensor)
#     # # print(explanation)
#     # with torch.no_grad():
#     #     result = run_func(basic_stream_tensor)
#     # result = result.cpu().numpy()
#     # result = Bitstream(result)
#     # global layer_counter
#     # if cfg_print_bitstream:
#     #     global_timer.append_data("run_regex", "inst_count", layer_counter)
#     # layer_counter = 0
#     # input_size = basic_stream_tensor.numel() / 1024 / 1024  # MB
#     # global_timer.append_data("run_regex", "input_size", input_size)
#     # avg_duration = global_timer.get_value("run_regex", "avg_duration")
#     # throughput = input_size / avg_duration * 1024  # MB/s
#     # global_timer.append_data("run_regex", "throughput", throughput)
#     # global_timer.append_data("run_regex", "count", result.get_count())
#     # return result
