#!/bin/bash
set -euo pipefail

wget https://hkustgz-my.sharepoint.com/:u:/g/personal/tge601_connect_hkust-gz_edu_cn/ES7vHG6o711Pp9Bpj2tr5hEB-RLa_ygdGbYIjxY6MT4spQ?e=iL6Hef\?e\=5bWc4W\&download=1 -O datasets_bitstream.tar.gz
mkdir -p datasets && tar -xzvf datasets_bitstream.tar.gz -C datasets