import argparse


def str2prce(str):
    prce = "/" + str + "/"
    return prce


def file2pcre(file, prce_file_path):
    with open(file, "r") as f:
        lines = f.readlines()
    pcre_lines = []
    for line in lines:
        pcre_lines.append(str2prce(line.replace("\n", "")) + "\n")
    with open(prce_file_path, "w") as f:
        f.writelines(pcre_lines)
    return prce_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-o", "--prce_file_path", type=str, required=True)
    args = parser.parse_args()

    file2pcre(args.file, args.prce_file_path)
    print("PCRE file is saved at: ", args.prce_file_path)
