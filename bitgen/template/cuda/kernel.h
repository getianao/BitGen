#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

extern "C" void CubExclusiveScanKernel(at::Tensor input);

extern "C" void byte_stream_transpose(at::Tensor bytestream,
                                      at::Tensor bitstream);