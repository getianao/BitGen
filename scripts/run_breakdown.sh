#!/bin/bash
set -euo pipefail

fullpath=$(readlink --canonicalize --no-newline $BASH_SOURCE)
cur_dir=$(
  cd $(dirname ${fullpath})
  pwd
)
# echo ${cur_dir}
if [ -z "$BITGEN_ROOT" ]; then
  echo "Error: BITGEN_ROOT environment variable is not defined."
  exit 1
fi

# export LOG=DEBUG

# Breakdown
SAVE_PATH=${BITGEN_ROOT}/results/csv/exec_opt_breakdown_arpg_1input.csv ${BITGEN_ROOT}/scripts/run_config.sh \
  ${BITGEN_ROOT}/configs/app/full/app_full.yaml \
  ${BITGEN_ROOT}/configs/exec/exec_opt_breakdown_arpg_1input.yaml
