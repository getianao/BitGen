import argparse
from bitgen.tool.pcre import read_regex_from_pcre_file
import yaml


def gen_app_str(regex_file_path, inputstream, regex_group_size, output, app_name):
    regexes = read_regex_from_pcre_file(regex_file_path)
    # group regexes
    regexes = [
        regexes[i : i + regex_group_size]
        for i in range(0, len(regexes), regex_group_size)
    ]
    # print(f"regexes: {regexes}")
    config = {
        "config_name": f"app_{app_name}",
        "root": "${BITGEN_ROOT}/datasets/Regex/",
        "apps": [],
    }
    for regex_id, regex in enumerate(regexes):
        config["apps"].append(
            {
                "name": f"{app_name}_str_{regex_id}",
                "type": "str",
                "regex": regex,
                "input": inputstream,
            }
        )
    # Write the data to a YAML file
    # print(f"config: {config}")
    with open(output, "w") as file:
        yaml.dump(
            config, file, default_flow_style=False, sort_keys=False, width=float("inf")
        )
    print("YAML file generated successfully!")
    print(f"Output file: {output}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate application string")
    arg_parser.add_argument("--regex-file-path", type=str, required=True)
    arg_parser.add_argument("--inputstream", type=str, required=True)
    arg_parser.add_argument("-o", "--output", type=str, required=True)
    arg_parser.add_argument(
        "--group-size", type=int, default=1, help="Number of regexes per group"
    )
    arg_parser.add_argument(
        "--app-name", type=str, required=True, help="Application name"
    )
    args = arg_parser.parse_args()
    regex_file_path = args.regex_file_path
    print(f"regex_file_path: {regex_file_path}")
    gen_app_str(regex_file_path, args.inputstream, args.group_size, args.output, args.app_name)

# python ${BITGEN_ROOT}/scripts/config/gen_app_str.py --group-size=16 --regex-file-path=${BITGEN_ROOT}/datasets/Regex/Bro217/regex/bro217.re.regex -o=${BITGEN_ROOT}/configs/app/app_bro_str_16regex.yaml
