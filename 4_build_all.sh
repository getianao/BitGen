#!/bin/bash
set -euo pipefail

cd ${BITGEN_ROOT}

pip install .  # BitGen python package
./scripts/install_icgrep.sh
./scripts/install_hs.sh
./scripts/install_ngap.sh