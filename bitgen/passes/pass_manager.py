from ..log import MyLogger

class PassManager:
    def __init__(
        self,
    ):
        self.passes = []

    def add_pass(self, pass_, enabled=True):
        if enabled:
            self.passes.append(pass_)

    def print_passes(self):
        for pass_ in self.passes:
            print(pass_.pass_name)

    def run(self, insts, var_name_map):
        for pass_ in self.passes:
            MyLogger.debug(f"[Pass] Running pass {pass_.pass_name}")
            insts, var_name_map = pass_.run(insts, var_name_map)
        return insts, var_name_map
