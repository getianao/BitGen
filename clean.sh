#!/bin/bash
fullpath=$(readlink --canonicalize --no-newline $BASH_SOURCE)
cur_dir=$(cd `dirname ${fullpath}`; pwd)
# echo ${cur_dir}

if [ -z "$BITGEN_ROOT" ]; then
  echo "Error: BITGEN_ROOT environment variable is not defined."
  exit 1
fi

rm -rf "${BITGEN_ROOT}/log/"
rm -rf "${BITGEN_ROOT}/.cache_bitgen/"