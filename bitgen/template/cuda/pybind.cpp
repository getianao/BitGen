#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "kernel.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cub_exclusive_scan", &CubExclusiveScanKernel);
  m.def("byte_stream_transpose", &byte_stream_transpose);
}