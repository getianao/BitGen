import argparse
import os
from tqdm import tqdm

from bitgen.tool import pcre2pablo
from bitgen.tool.pcre2pablo import run_icgrep, run_icgrep_from_file
from bitgen.tool.pcre import read_regex_from_pcre_file


def filter_unique(pcre_file_path, filtered_pcre_file_path):
    regexes = read_regex_from_pcre_file(pcre_file_path)
    regexes_filtered = []
    for regex in tqdm(regexes):
        if regex not in regexes_filtered:
            regexes_filtered.append(regex)
    print("Total PCREs number: ", len(regexes))
    print("Filtered PCREs number: ", len(regexes_filtered))
    print("Removed PCREs number: ", len(regexes) - len(regexes_filtered))
    with open(filtered_pcre_file_path, "w") as f:
        for regex_f in regexes_filtered:
            f.write("/" + regex_f + "/\n")
        
    
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, required=True, help="Pcre file path")
    parser.add_argument("-o", type=str, required=False, help="Filtered pcre file path")
    args = parser.parse_args()

    if args.o is None:
        pathname, ext = os.path.splitext(args.f)
        args.o = pathname + ".unique" + ext
        print("Set output file path to: ", args.o)

    filter_unique(args.f, args.o)
    print("Filtered PCRE file is saved at: ", args.o)
