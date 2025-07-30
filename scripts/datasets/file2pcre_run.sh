#!/bin/bash
fullpath=$(readlink --canonicalize --no-newline $BASH_SOURCE)
cur_dir=$(cd `dirname ${fullpath}`; pwd)
# echo ${cur_dir}

RULESET_ROOT="$BITGEN_ROOT/datasets/Regex"

for ruleset in $(ls $RULESET_ROOT)
do
    # Exclude the directory
    # if [ -d $RULESET_ROOT/$ruleset ]; then
    #     continue
    # fi
    regex_file_folder=$RULESET_ROOT/$ruleset/regex
    regex_file=$(find $regex_file_folder -type f | head -n 1)
    echo $regex_file
    python $BITGEN_ROOT/scripts/datasets/file2pcre.py -f=$regex_file -o=$regex_file.regex
done

