#!/bin/bash
set -euo pipefail
set -x

# Ensure BITGEN_ROOT is set
if [[ -z "${BITGEN_ROOT:-}" ]]; then
  echo "Error: BITGEN_ROOT environment variable is not set."
  exit 1
fi
cd "$BITGEN_ROOT"

if [[ -d "parabix-devel" ]]; then
  rm -rf parabix-devel
fi
git clone -b icgrep-1.0 https://github.com/getianao/parabix-devel.git
cd parabix-devel

# LLVM-3.5
if [[ -d "llvm-project" ]]; then
  rm -rf llvm-project
fi
git clone -b llvmorg-3.5.0 --depth 1 https://github.com/llvm/llvm-project.git
mkdir -p libllvm
cd llvm-project && mkdir -p build && cd build

cmake -DCMAKE_INSTALL_PREFIX=../../libllvm \
      -DLLVM_TARGETS_TO_BUILD=X86 \
      -DLLVM_BUILD_TOOLS=OFF \
      -DLLVM_BUILD_EXAMPLES=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      ../llvm
make -j"$(nproc)"
make install

cd $BITGEN_ROOT/parabix-devel && mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=../libllvm -DCMAKE_BUILD_TYPE=Release ../icgrep-1.00
make -j"$(nproc)"