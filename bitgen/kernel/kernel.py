from typing import List, Optional
import hashlib

from ..nfa import NFA, State
from .. import bitstream as bs

class kernel_bs:
    bs_result = None

    def match_state(self, input_stream, state: State):
        # print(f"state: {state.symbol_set_expr}")
        bitstream = bs.create_zeros(len(input_stream) + 1)
        for i, symbol in enumerate(input_stream):
            # print(f"symbol: {symbol}, state.symbol_set: {state.symbol_set_expr}, state.match(symbol): {state.match(symbol)}")
            bitstream.set_bit(i, state.match(symbol))
        return bitstream

    # Match input stream from the state
    def match(
        self,
        input_stream,
        start_states: List[State],
        bitstreams: List[bs.Bitstream] = None,
    ):
        states = start_states
        bitstreams = bitstreams
        states_next = []
        bitstreams_next = []

        while len(states) > 0:
            print(f"len(states): {len(states)}")
            for state, bitstream in zip(states, bitstreams):
                count =  "None" if bitstream is None else bitstream.get_count()
                is_zero = "None" if bitstream is None else bitstream.is_zero()
                # if bitstream is not None and not bitstream.is_zero():
                #     print(f"state: {state.symbol_set_expr}, count: {count}, is_zero: {is_zero}, pos: {bitstream.get_ones_pos()}")
                if bitstream is None:
                    bitstream = bs.create_ones(len(input_stream) + 1)
                if bitstream.is_zero():
                    continue
                cc_stream = self.match_state(input_stream, state)
                bitstream = bs.bitwise_and(bitstream, cc_stream)
                bitstream = bs.bitwise_shift_right(bitstream, 1)
                if state.is_report():
                    self.bs_result = bs.bitwise_or(self.bs_result, bitstream)
                if len(state.neighbors) == 0:
                    continue
                states_next += state.neighbors
                bitstreams_next += [bitstream] * len(state.neighbors)
            # states, bitstreams = self.unique(states_next, bitstreams_next)
            states, bitstreams = states_next, bitstreams_next
            states_next = []
            bitstreams_next = []
        return

    def run(self, nfa: NFA, input_stream: str):
        self.bs_result = bs.create_zeros(len(input_stream) + 1)
        all_start_states = nfa.get_always_active_states()
        bitstreams = [None] * len(all_start_states)
        all_start_states += nfa.get_start_states()
        start_bitstreams = bs.create_zeros(len(input_stream) + 1)
        start_bitstreams.set_bit(0, 1)
        bitstreams += [start_bitstreams] * len(nfa.get_start_states())
        self.match(input_stream, all_start_states, bitstreams)
        return self.bs_result
