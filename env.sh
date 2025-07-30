#!/bin/bash
fullpath=$(readlink --canonicalize --no-newline $BASH_SOURCE)
cur_dir=$(cd `dirname ${fullpath}`; pwd)
# echo ${cur_dir}

export PYTHONUNBUFFERED=1

export BITGEN_ROOT=${cur_dir}
export PATH=${BITGEN_ROOT}/parabix-devel/build:$PATH             # icgrep
export PATH=${BITGEN_ROOT}/hscompile/build:$PATH             # hscompile
export PATH=${BITGEN_ROOT}/ngAP/code/build/bin:$PATH     # ngap
export PATH=${BITGEN_ROOT}/VASim:$PATH                     # VASim

export CUDA_HOME=$(dirname $(dirname $(command -v nvcc)))
# Fix undefined symbol: __nvJitLinkAddData_12_4
# export LD_LIBRARY_PATH=$(spack location -i cuda@12.4.1)/lib64:$LD_LIBRARY_PATH

# sudo nvidia-smi -pm 1