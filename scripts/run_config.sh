#!/bin/bash
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

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <app_config> <exec_config> [additional_arguments...]"
  exit 1
fi

APP_PATH="$1"
EXEC_PATH="$2"

ADDITIONAL_ARGS="${@:3}"

LOG_DIR="${BITGEN_ROOT}/log"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ "${PRFILE_CUDA:-0}" -eq 1 ]; then
  CMD=(time python -u "${BITGEN_ROOT}/scripts/run_config.py" --app="${APP_PATH}" --exec="${EXEC_PATH}" --profile --options=\"${ADDITIONAL_ARGS}\")
else
  CMD=(time python -u "${BITGEN_ROOT}/scripts/run_config.py" --app="${APP_PATH}" --exec="${EXEC_PATH}" --options=\"${ADDITIONAL_ARGS}\")
fi


LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
echo "Executing command: ${CMD[@]}" | tee "$LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
if [ "${PIPESTATUS[0]}" -eq 0 ]; then
  RESULT_FILE="$(tail -n 10 "$LOG_FILE" | grep "Data saved to" | awk '{print $4}')"
  if [ -n "$SAVE_PATH" ]; then # SAVE_PATH is not empty
    if [ -e "$SAVE_PATH" ]; then  # SAVE_PATH already exists
      BACKUP_PATH="${SAVE_PATH}.bak"
      echo "Backup existing save path to ${BACKUP_PATH}" | tee -a "$LOG_FILE"
      mv "$SAVE_PATH" "$BACKUP_PATH"
    fi
    SAVE_DIR=$(dirname "$SAVE_PATH")
    mkdir -p "$SAVE_DIR"
    cp "$RESULT_FILE" "$SAVE_PATH"
    echo
    echo "Copied result from ${RESULT_FILE} to ${SAVE_PATH}" | tee -a "$LOG_FILE"
  fi
else
  echo
  echo "Command failed. Result file will not be saved." | tee -a "$LOG_FILE"
fi
echo "Log file saved to: ${LOG_FILE}"
echo "=============================================== run_config.sh end "
