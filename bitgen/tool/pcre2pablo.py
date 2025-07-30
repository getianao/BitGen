import argparse
import os
import sys
import subprocess
import shlex
from colorama import Fore, Style
from .command_tool import exe_command


cuurent_dir = os.path.dirname(os.path.abspath(__file__))
small_input = os.path.join(cuurent_dir, "../datasets_small/input/abcde.txt")


def pcre2pablo(pcre, pablo):

    icgrep_cmd = [
        "icgrep",
        # '-e="' + pcre + '"',
        "-e=" + pcre,
        f"-print-pablo={pablo}",
        "-c",
        "-if-insertion-gap=2147483647",
        "-disable-matchstar",
        "-disable-Unicode-linebreak",
        # "-disable-log2-bounded-repetition",
        # "-disable-Unicode-matchstar",
        small_input,
    ]
    exe_command(icgrep_cmd, check=False, shell=False)
    return pablo


def pcrefile2pablo(pcre_file_path, pablo):
    icgrep_cmd = [
        "icgrep",
        # '-e="' + pcre + '"',
        "-f=" + pcre_file_path,
        f"-print-pablo={pablo}",
        "-c",
        "-if-insertion-gap=2147483647",
        "-disable-matchstar",
        "-disable-Unicode-linebreak",
        # "-disable-log2-bounded-repetition",
        # "-disable-Unicode-matchstar",
        small_input,
    ]
    output = exe_command(icgrep_cmd, check=False, shell=False)
    return output


def run_icgrep(pcre, text, repeat=1, input_size=-1, verbose=True):
    icgrep_cmd = [
        "icgrep",
        "-e=" + pcre,
        "-c",
        # "-if-insertion-gap=2147483647",
        # "-disable-matchstar",
        # "-disable-Unicode-linebreak",
        # "-disable-log2-bounded-repetition",
        # "-disable-Unicode-matchstar",
        f"-d={repeat}",
        f"-s={input_size}",
        text,
    ]
    output = exe_command(icgrep_cmd, check=False, shell=False, verbose=verbose)
    return output


def run_icgrep_from_file(file_path, text, repeat=1, input_size=-1, verbose=True):
    icgrep_cmd = [
        "icgrep",
        "-f=" + file_path,
        "-c",
        # "-if-insertion-gap=2147483647",
        # "-disable-matchstar",
        # "-disable-Unicode-linebreak",
        # "-disable-log2-bounded-repetition",
        # "-disable-Unicode-matchstar",
        f"-d={repeat}",
        f"-s={input_size}",
        text,
    ]
    output = exe_command(icgrep_cmd, check=False, shell=False, verbose=verbose)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PCRE to Pablo")
    args = parser.parse_args()
