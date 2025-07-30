
#!/bin/bash
set -euo pipefail
set -x

# Ensure BITGEN_ROOT is set
if [[ -z "${BITGEN_ROOT:-}" ]]; then
  echo "Error: BITGEN_ROOT environment variable is not set."
  exit 1
fi
cd "$BITGEN_ROOT"

if [[ -d "hscompile" ]]; then
  rm -rf hscompile
fi
# commit: ae44acb94e9018a0ecd09b9aa1c30a4e7d7ccf99
git clone https://github.com/getianao/hscompile.git
cd hscompile && mkdir -p lib && cd lib

# Hyperscan v4.4.1
if [[ -d "hyperscan" ]]; then
  rm -rf hyperscan
fi
git clone -b v4.4.1 https://github.com/intel/hyperscan.git
cd hyperscan
# https://github.com/intel/hyperscan/issues/292
awk 'NR==18 {$0="nm -f p -g -D ${LIBC_SO} | sed -s '\''s/\\([^ @]*\\).*/^\\1$/'\'' >> ${KEEPSYMS}"} {print}' ./cmake/build_wrapper.sh > temp && mv temp ./cmake/build_wrapper.sh
chmod +x ./cmake/build_wrapper.sh
awk 'NR==183 {print "        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS GNUCXX_MINVER)"; next} {print}' ./CMakeLists.txt > temp && mv temp ./CMakeLists.txt
sed -i '/#include <mutex>/a #include <stdexcept>' ./tools/hsbench/thread_barrier.h
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc ..
make -j"$(nproc)"

# MNRL
cd $BITGEN_ROOT/hscompile/lib
if [[ -d "mnrl" ]]; then
  rm -rf mnrl
fi
git clone -b v1.0 https://github.com/kevinaangstadt/mnrl
cd mnrl/C++

sed -i 's/^CC = .*/CC = g++-5/' Makefile  # requires GCC-5.
make  # If an error occurs, try to run it again
cd $BITGEN_ROOT/hscompile && mkdir -p build && cd build

cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc \
      -DHS_SOURCE_DIR=../lib/hyperscan \
      -DMNRL_SOURCE_DIR=../lib/mnrl/C++ \
      ..
make -j"$(nproc)"
