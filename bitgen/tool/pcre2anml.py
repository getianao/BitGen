import argparse
import os
import sys
import subprocess
import shlex
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


def pcre2anml(pcre, anml=None, regex_num=-1, random_select=False):
    # bs_home_path = os.environ.get("BS_HOME")
    # if bs_home_path is None:
    #     print("BS_HOME environment variable not set.")
    #     exit(1)
    # # vasim will generate the ANML file in the current directory
    # os.chdir(bs_home_path + "/bitstream/python/tool")

    if not os.path.isfile(pcre):
        pcre_expr = pcre
        pcre = "/tmp/pcre2anml.pcre"
        text_cmd = f"echo /'{pcre_expr}'/ > {pcre}"
        exe_command(text_cmd)

    if not anml:
        anml = "/tmp/" + os.path.basename(pcre) + ".anml"
        
    # Select regex
    prces = read_pcre_from_pcre_file(pcre)
    prces = select_regex(prces, regex_num, random_select)
    pcre_path2 = "/tmp/pcre2anml2.pcre"
    with open(pcre_path2, "w") as f:
        for r_id, r in enumerate(prces):
            if r_id != 0:
                f.write("\n")
            f.write(r)

    anml_tmp = "automata_0.anml"
    mnrl = "/tmp/" + os.path.basename(pcre_path2) + ".mnrl"
    pcre2mnrl = f"pcre2mnrl {pcre_path2} {mnrl}"
    mnrl2anml = f"vasim -a {mnrl}"
    mv = f"mv {anml_tmp} {anml}"

    result = exe_command(pcre2mnrl)
    if result.returncode != 0 or result.stderr:
        raise Exception(f"Failed to convert PCRE to MNRL: {result.stderr}")
    result = exe_command(mnrl2anml)
    if result.returncode != 0 or result.stderr:
        raise Exception(f"Failed to convert MNRL to ANML: {result.stderr}")
    exe_command(mv)
    # print(f"ANML generated: {anml}")
    return anml


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert PCRE to ANML')
    parser.add_argument('-e', required=True, help='PCRE file (with slashes) or expression (without slashes)')
    parser.add_argument('-o', help='ANML file')
    
    args = parser.parse_args()
    pcre2anml(args.e, args.o)
