#!/bin/bash
set -euo pipefail

docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t bitgen-ae ./docker