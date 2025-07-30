#!/bin/bash
set -euo pipefail

cd ${BITGEN_ROOT}

./scripts/run_throughput.sh
./scripts/run_breakdown.sh