import argparse
import os
from tqdm import tqdm

from bitgen.tool import pcre2pablo
from bitgen.tool.pcre2anml import pcre2anml
from bitgen.tool.pcre import read_regex_from_pcre_file


def filter_pcre_icgrep(pcre_file_path, filtered_pcre_file_path):
    regexes = read_regex_from_pcre_file(pcre_file_path)
    print("Total PCREs number: ", len(regexes))
    dummy_input = pcre2pablo.small_input
    regexes_filtered = []
    with open(filtered_pcre_file_path, "w") as f:
        for regex in tqdm(regexes):
            # regex = regex.strip()
            # print("Processing: ", regex)
            tmp_regex_file = "/tmp/tmp_regex_filter.txt"
            with open(tmp_regex_file, "w") as f_tmp:
                f_tmp.write("/" + regex + "/")
            
            try:
                anml_path = pcre2anml(tmp_regex_file)
            except Exception as e:
                print(f"PCRE /{regex}/ is filtered out. Error: {e}")
                continue
            regexes_filtered.append(regex)

        print("Original PCREs number: ", len(regexes))
        print("Filtered PCREs number: ", len(regexes_filtered))
        print("Removed PCREs number: ", len(regexes) - len(regexes_filtered))
        for regex_f in regexes_filtered:
            f.write("/" + regex_f + "/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, required=True, help="Pcre file path")
    parser.add_argument("-o", type=str, required=False, help="Filtered pcre file path")
    args = parser.parse_args()

    if args.o is None:
        pathname, ext = os.path.splitext(args.f)
        args.o = pathname + ".anml" + ext
        print("Set output file path to: ", args.o)

    filter_pcre_icgrep(args.f, args.o)
    print("Filtered PCRE file is saved at: ", args.o)
