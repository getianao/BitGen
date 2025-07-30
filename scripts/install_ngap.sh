#!/bin/bash
set -euo pipefail
set -x

# Ensure BITGEN_ROOT is set
if [[ -z "${BITGEN_ROOT:-}" ]]; then
  echo "Error: BITGEN_ROOT environment variable is not set."
  exit 1
fi
cd "$BITGEN_ROOT"

if [[ -d "VASim" ]]; then
  rm -rf VASim
fi
# 3a9781bf33909166825b27506da339f985140416
git clone https://github.com/jackwadden/VASim.git
cd VASim
sed -i '/#include <bitset>/a #include <cstdint>' ./include/util.h
sed -i '52s|$(MAKE) $(LIBPUGI)|$(MAKE) $(LIBPUGI) CXXFLAGS+=" -Wno-self-move"|' Makefile
make

cd $BITGEN_ROOT
if [[ -d "ngAP" ]]; then
  rm -rf ngAP
fi
git clone -b loop https://github.com/getianao/ngAP.git
echo "Ensure the GPU architecture in ngAP/code/CMakeLists.txt matches your hardware (e.g., sm_86 for NVIDIA A100)."
# sed -i '29s|#define DATA_BUFFER_SIZE 1000000000LL.*|#define DATA_BUFFER_SIZE 300000000         // 1.2GB|' ngAP/code/src/ngap/ngap_buffer.h
cd ngAP/code && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j