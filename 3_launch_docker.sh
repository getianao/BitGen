#!/bin/bash
set -euo pipefail

docker run -it --rm --gpus all -v ./:/BitGen bitgen-ae:latest /bin/bash