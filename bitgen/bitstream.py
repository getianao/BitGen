import numpy as np
import textwrap
import torch
import math

from . import config as cfg
from .cuda import byte_stream_transpose
from .tool import global_timer


bs_debug = False

def swap_endian(value: np.uint32):
    # Ensure value is treated as 32-bit unsigned
    value &= 0xFFFFFFFF

    # Extract and shift each byte
    byte0 = (value & 0x000000FF) << 24
    byte1 = (value & 0x0000FF00) << 8
    byte2 = (value & 0x00FF0000) >> 8
    byte3 = (value & 0xFF000000) >> 24

    # Combine the shifted bytes
    swapped = byte0 | byte1 | byte2 | byte3
    return swapped

class Bitstream:
    data = None

    def __init__(self, data: np.ndarray = None):
        if isinstance(data, np.ndarray):
            input_shape = list(data.shape)
            if data is not None and data.dtype != bool:
                if data.dtype == np.uint32:
                    for i in range(len(data)):
                        data[i] = swap_endian(data[i])
                    arr_bytes = data.view(np.uint8)
                    bits = np.unpackbits(arr_bytes, bitorder="big")
                    # bits_reshaped = bits.reshape(-1, 32)
                    bits_bool = bits.astype(bool)
                elif data.dtype == np.uint8:
                    bits = np.unpackbits(data, bitorder="big")
                    bits_bool = bits.astype(bool)
                else:
                    raise Exception(f"Invalid data type: {data.dtype}")
            data = bits_bool.flatten()
            input_shape[-1] = -1
            data = data.reshape(input_shape)
        else:
            raise Exception("Invalid data type")

        self.data = data

    def set_bit(self, i, value):
        self.data[i] = value

    def get_bit(self, i):
        return self.data[i]

    def get_count(self):
        return np.count_nonzero(self.data)

    def get_count_array(self):
        assert self.data.ndim == 4
        non_zero_count = np.count_nonzero(self.data, axis=-1)
        result = np.sum(non_zero_count, axis=(0, 1))
        return result

    def get_nonzero_positions(self):
        nonzero_indices = np.nonzero(self.data)
        count = len(nonzero_indices[0])  # The number of nonzero elements
        positions = list(zip(*nonzero_indices))  # Combine indices into tuples
        return positions

    def get_ones_pos(self):
        return np.where(self.data == 1)[0]

    def is_zero(self):
        return not np.any(self.data)

    def save(self, filename):
        s = self.__str__(size = len(self.data))
        with open(filename, 'w') as f:
            f.write(s)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.data, other.data)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self, size=64):
        data = self.data.flatten()
        binary_string = ''.join(str(int(b)) for b in data[:size])
        wrapped = textwrap.wrap(binary_string, 8)
        formatted_binary = " ".join(wrapped)
        aligned = f"{formatted_binary: <10}"
        return f'{aligned}'

    def __repr__(self):
        return f'Bitstream(data=[{len(self.data)}])'


def create_zeros(n):
    bs = Bitstream(np.zeros(n, dtype=bool))
    return bs

def create_ones(n):
    bs = Bitstream(np.ones(n, dtype=bool))
    return bs


# Operation
def bitwise_and(bs_1 : Bitstream, bs_2 : Bitstream):
    data = np.logical_and(bs_1.data, bs_2.data)
    if bs_debug:
        print(f'bitwise_and:')
        __print_aligned('bs_1:', bs_1)
        __print_aligned('bs_2:', bs_2)
        __print_aligned('result:', Bitstream(data))
    return Bitstream(data)

def bitwise_or(bs_1 : Bitstream, bs_2 : Bitstream):
    data = np.logical_or(bs_1.data, bs_2.data)
    if bs_debug:
        print(f'bitwise_or:')
        __print_aligned('bs_1:', bs_1)
        __print_aligned('bs_2:', bs_2)
        __print_aligned('result:', Bitstream(data))
    return Bitstream(data)

def bitwise_xor(bs_1 : Bitstream, bs_2 : Bitstream):
    data = np.logical_xor(bs_1.data, bs_2.data)
    if bs_debug:
        print(f'bitwise_xor:')
        __print_aligned('bs_1:', bs_1)
        __print_aligned('bs_2:', bs_2)
        __print_aligned('result:', Bitstream(data))
    return Bitstream(data)

def bitwise_not(bs : Bitstream):
    data = np.logical_not(bs.data)
    if bs_debug:
        print(f'bitwise_not:')
        __print_aligned('bs:', bs)
        __print_aligned('result:', Bitstream(data))
    return Bitstream(data)

def bitwise_shift_right(bs : Bitstream, n):
    data = np.roll(bs.data, n)
    data[:n] = 0
    if bs_debug:
        print(f'bitwise_shift_right:')
        __print_aligned('bs:', bs)
        __print_aligned('result:', Bitstream(data))
    return Bitstream(data)


def __print_aligned(msg, bs):
    print(f'{msg:<8} {bs}')


def transpose_bytestream_to_bitstream(
    input_stream_tensor: torch.Tensor,
) -> torch.Tensor:
    # input_stream_tensor = input_stream_tensor.contiguous()
    assert input_stream_tensor.dim() == 3
    unit_size = 32
    n_char = input_stream_tensor.size(-1)
    n_unit = math.ceil(n_char / unit_size)
    padding = (n_unit * unit_size) - n_char
    if padding > 0:
        input_padded = torch.nn.functional.pad(
            input_stream_tensor, (0, padding), "constant", 0
        )
    else:
        input_padded = input_stream_tensor
    # CUDA or python implementation
    if True:
        input_padded = input_padded.cuda().contiguous()
        bitstreams = (
            torch.zeros(
                input_padded.size(0),
                input_padded.size(1),
                8,
                n_unit,
                dtype=torch.uint32,
            )
            .cuda()
            .contiguous()
        )
        # print("input_padded", input_padded.shape)
        # print("bitstreams", bitstreams.shape)
        # Warmup
        for i in range(8):
            byte_stream_transpose(input_padded, bitstreams)
        # Run
        for i in range(8):
            with global_timer.time("transpose"):
                byte_stream_transpose(input_padded, bitstreams)

        if cfg.get_config("backend") == "torch":
            bitstreams = bitstreams.view(torch.uint8)
            bitstreams_shape = bitstreams.shape
            bitstreams = bitstreams.view(
                bitstreams_shape[:-1] + (bitstreams_shape[-1] // 4, 4)
            )
            bitstreams = bitstreams.flip(-1).view(bitstreams_shape)
    else:
        # Warmup
        for i in range(4):
            bitstreams = transpose_bytestream_to_bitstream_py(input_padded)
        # Run
        for i in range(8):
            with global_timer.time("transpose"):
                bitstreams = transpose_bytestream_to_bitstream_py(input_padded)
    input_size = bitstreams.numel() * bitstreams.element_size() / 1024 / 1024  # MB
    global_timer.append_data("transpose", "input_size", input_size)
    avg_duration = global_timer.get_value("transpose", "avg_duration")
    throughput = input_size / avg_duration * 1024  # MB/s
    global_timer.append_data("transpose", "throughput", throughput)
    return bitstreams


def transpose_bytestream_to_bitstream_py(
    input_padded: torch.Tensor,
) -> torch.Tensor:
    input_np = input_padded.cpu().numpy()
    bits = np.unpackbits(input_np)
    bits = bits.reshape(*input_np.shape, 8)
    if bits.ndim == 2:
        bits = bits.transpose(1, 0)  # Shape: (8, n_bytes)
    elif bits.ndim == 3:
        bits = bits.transpose(0, 2, 1)  # Shape: (split, 8, n_bytes)
    elif bits.ndim == 4:
        bits = bits.transpose(0, 1, 3, 2)  # Shape: (split, multi, 8, n_bytes)
    assert bits.shape[-1] % 8 == 0
    bitstreams_np = np.packbits(bits, axis=-1, bitorder="big")
    # print("bitstreams_np", bitstreams_np)
    # print("bitstreams_np", bitstreams_np.shape)
    # TODO(tge): refactor it
    if cfg.get_config("backend") == "cuda":
        # uint8 to uint32
        assert bitstreams_np.shape[-1] % 4 == 0
        bitstreams_np_u32 = bitstreams_np.reshape(
            bitstreams_np.shape[:-1] + (bitstreams_np.shape[-1] // 4, 4)
        ).astype(np.uint32)
        bitstreams_np_u32_2 = np.zeros(bitstreams_np_u32.shape[:-1], dtype=np.uint32)
        for i in range(4):
            bitstreams_np_u32_2 += bitstreams_np_u32[..., i] << (24 - i * 8)
        # print("bitstreams_np_u32_2", bitstreams_np_u32_2)
        # print("bitstreams_np_u32_2", bitstreams_np_u32_2.shape)
        bitstreams_np = bitstreams_np_u32_2
    bitstreams = torch.from_numpy(bitstreams_np).cuda()
    bitstreams = bitstreams.contiguous()
    return bitstreams
