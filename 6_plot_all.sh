#!/bin/bash
set -euo pipefail

cd ${BITGEN_ROOT}

python ./scripts/plot/plot_app_full_new.py
python ./scripts/plot/plot_app_full_breakdown.py