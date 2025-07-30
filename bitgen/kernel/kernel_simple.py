from ..nfa import NFA, State
from .. import bitstream as bs

class kernel_simple:
    bs_result = None

    def run(self, nfa: NFA, input_stream: str) -> bs.Bitstream:
        self.bs_result = bs.create_zeros(len(input_stream) + 1)

        start_states = nfa.get_start_states()
        always_active_states = nfa.get_always_active_states()
        worklist = []
        worklist_next = []
        worklist += start_states
        worklist += always_active_states

        # print(f"start_states: {start_states}")
        # print(f"always_active_states: {always_active_states}")

        for symbol_idx, symbol in enumerate(input_stream):
            for state in worklist:
                if state.match(symbol):
                    # print(f"state.neighbors: {state.neighbors}")
                    worklist_next+=state.neighbors
                    if state.is_report():
                        self.bs_result.set_bit(symbol_idx + 1, 1)
            worklist = worklist_next
            worklist += always_active_states
            worklist_next = []
        return self.bs_result
