import re
import numpy as np
import textwrap
import inspect

# bs_debug = True
bs_debug = False

def print_callee_line_number():
    caller_frame = inspect.currentframe().f_back.f_back
    line_number = caller_frame.f_lineno
    return line_number

def print_np_as_bs(data: np.ndarray):
    binary_string = ''.join(str(int(b)) for b in data[:64])
    wrapped = textwrap.wrap(binary_string, 8)
    formatted_binary = " ".join(wrapped)
    aligned = f"{formatted_binary: <10}"
    return print(f'{aligned}')

def bs_match(input_stream: str, symbol_set_expr: str):
    result = np.zeros(len(input_stream) + 1, dtype=bool)
    for i, symbol in enumerate(input_stream):
        result[i] = bool(re.fullmatch(symbol_set_expr, symbol))
    return result

def bs_and(a : np.ndarray, b : np.ndarray):
    r = np.logical_and(a, b)
    if bs_debug:
        print(f'bs_and:')
        print_np_as_bs(a)
        print_np_as_bs(b)
        print_np_as_bs(r)
    return r

def bs_or(a : np.ndarray, b : np.ndarray):
    r = np.logical_or(a, b)
    if bs_debug:
        print(f'bs_or:')
        print_np_as_bs(a)
        print_np_as_bs(b)
        print_np_as_bs(r)
    return r


def bs_xor(a : np.ndarray, b : np.ndarray):
    r = np.logical_xor(a, b)
    if bs_debug:
        print(f'bs_xor:')
        print_np_as_bs(a)
        print_np_as_bs(b)
        print_np_as_bs(r)
    return r

def bs_not(a : np.ndarray):
    r = np.logical_not(a)
    if bs_debug:
        print(f'bs_not:')
        print_np_as_bs(a)
        print_np_as_bs(r)
    return r

def bs_advance(a : np.ndarray, n):
    r = np.zeros(len(a), dtype=bool)
    r[n:] = a[:-n]
    if bs_debug:
        print(f'bs_advance:')
        print_np_as_bs(a)
        print_np_as_bs(r)
    return r


def bs_full_adder(a: bool, b: bool, c: bool):
    # FullAdder(A, B, C) = (A ⊕ B ⊕ C, (A ∧ B) ∨ (C ∧ (A ⊕ B)))
    s = a ^ b ^ c
    carry = (a & b) | (c & (a ^ b))
    return s, carry


def bs_add(a: np.ndarray, b: np.ndarray):
    assert len(a) == len(b)
    carry = False
    for i in range(len(a)):
        a[i], carry = bs_full_adder(a[i], b[i], carry)
    return a


def bs_scan_thru(a: np.ndarray, b: np.ndarray):
    # ScanThru(M, C) = (C + M) ∧ ~M
    m_plus_c = bs_add(a, b)
    not_m = bs_not(a)
    c_plus_m_and_not_m = bs_and(m_plus_c, not_m)
    return c_plus_m_and_not_m


def bs_match_star(a: np.ndarray, b: np.ndarray):
    # MatchStar(M, C) = (((M ∧ C) + C) ⊕ C) ∨ M
    m_and_c = bs_and(a, b)
    m_and_c_plus_c = bs_add(m_and_c, b)
    m_and_c_plus_c_xor_c = bs_xor(m_and_c_plus_c, b)
    m_and_c_plus_c_xor_c_or_m = bs_or(m_and_c_plus_c_xor_c, a)
    return m_and_c_plus_c_xor_c_or_m


def bs_is_zero(a : np.ndarray):
    if bs_debug:
        print(f'bs_is_zero:')
        print_np_as_bs(a)
    return not np.any(a)

def create_zeros(n):
    return np.zeros(n, dtype=bool)

def create_ones(n):
    return np.ones(n, dtype=bool)


def create_start(n):
    r = np.zeros(n, dtype=bool)
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

def transpose_byte_to_bitstream(input_stream: str):
    n_char = len(input_stream)
    basic_stream = np.zeros((8, n_char + 1), dtype=bool)
    # first bit to 8th bit
    for ch_id in range(n_char):
        ch = input_stream[ch_id]
        ch_ord = ord(ch)
        for bit_id in range(8):
            bit_value = bool(ch_ord & (1 << (7 - bit_id)))
            basic_stream[bit_id][ch_id] = bit_value
    return basic_stream
