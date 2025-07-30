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

cd ${BITGEN_ROOT}

# export LOG=DEBUG

# Throughput BitGen
SAVE_PATH=${BITGEN_ROOT}/results/csv/exec_tune_best_arpg_1input.csv ./scripts/run_config.sh \
  ${BITGEN_ROOT}/configs/app/full/app_full.yaml \
  ${BITGEN_ROOT}/configs/exec/exec_tune_best_arpg_1input.yaml

# Throughput ngap icgrep
SAVE_PATH=${BITGEN_ROOT}/results/csv/exec_baseline_arpg_1input.csv ./scripts/run_config.sh \
  ${BITGEN_ROOT}/configs/app/full/app_full.yaml \
  ${BITGEN_ROOT}/configs/exec/exec_baseline_arpg_1input.yaml

# Throughput hs
SAVE_PATH=${BITGEN_ROOT}/results/csv/exec_baseline_hs_arpg_1input.csv ./scripts/run_config.sh \
  ${BITGEN_ROOT}/configs/app/full/app_full.yaml \
  ${BITGEN_ROOT}/configs/exec/exec_baseline_hs_arpg_1input.yaml