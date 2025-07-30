import argparse
import os
import random
from colorama import Fore, Style
from .command_tool import exe_command

def select_regex(regexes, regex_num=-1, random_select=False):
    if regex_num > 0:
        if random_select:
            print(
                f"Random select {regex_num} regexes from file ({len(regexes)} regexes)"
            )
            random.seed(42)
            regexes = random.sample(regexes, regex_num)
        else:
            print(f"Select {regex_num} regexes from file ({len(regexes)} regexes)")
            regexes = regexes[:regex_num]
    return regexes


def read_pcre_from_pcre_file(pcre_file):
    pcres = []
    with open(pcre_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            regex = line.strip()
            if len(regex) > 0:
                pcres.append(regex)
    return pcres


def pcre2hs(pcre, suffix=None):
    if suffix is not None:
        pcre_path = f"/tmp/pcre2anml_{suffix}.pcre"
    else:
        pcre_path = "/tmp/pcre2anml.pcre"
    if isinstance(pcre, list):
        with open(pcre_path, "w") as f:
            for i, r in enumerate(pcre):
                if i != 0:
                    f.write("\n")
                f.write('/' + r + '/')
    elif not os.path.isfile(pcre):
        # regex = regex.replace("'", "'\\''")
        pcre_expr = pcre
        text_cmd = f"echo /'{pcre_expr}'/ > {pcre_path}"
        exe_command(text_cmd)
    else:
        pcre_path = pcre

    mnrl_path = os.path.join("/tmp/", os.path.basename(pcre_path) + ".mnrl")
    pcre2mnrl_command = f"pcre2mnrl {pcre_path} {mnrl_path}"
    hs_path = os.path.join("/tmp/", os.path.basename(pcre_path) + ".hs")
    mnrl2hs_command = f"hscompile {mnrl_path} {hs_path}"

    exe_command(pcre2mnrl_command)
    exe_command(mnrl2hs_command)
    return hs_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert PCRE to Haskell')
    parser.add_argument('-e', required=True, help='PCRE file (with slashes) or expression (without slashes)')
    args = parser.parse_args()
    pcre2hs(args.e)
