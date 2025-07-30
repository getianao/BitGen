import random

# PCRE reequires regex to be enclosed in slashes.
# regex is a string by removing the slashes from pcre.
def read_regex_from_pcre_file(regex_file):
    regexes = []
    with open(regex_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            regex = line.strip()
            if len(regex) == 0:
                continue
            #  Try remove slashes
            first_slash = regex.find("/")
            last_slash = regex.rfind("/")
            if (
                first_slash == 0
                # and last_slash == len(regex) - 1
                and first_slash != last_slash
                and last_slash > first_slash
            ):
                # It has slashes
                regex = regex[first_slash + 1 : last_slash]
            else:
                raise ValueError(f"Regex requires slashes: {regex}")
            regexes.append(regex)
    return regexes


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


def pcre2file(prce_file, file_path, regex_num=-1, random_select=False):
    prce = read_regex_from_pcre_file(prce_file)
    prce = select_regex(prce, regex_num, random_select)

    with open(file_path, 'w') as f:
        for r_id, r in enumerate(prce):
            if r_id != 0:
                f.write("\n")
            f.write(r)
    return file_path
