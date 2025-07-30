#include "kernel.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include <stdint.h>

inline void checkCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file
              << ":" << line << std::endl;
    exit(err);
  }
}

#define CUDA_CHECK(call) checkCudaError(call, __FILE__, __LINE__)

struct Overflow {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    // (a->carry & b->max) | (b->carry)
    return (((a & (b >> 1)) | b) & 1);
  }
};

extern "C" void CubExclusiveScanKernel(at::Tensor input) {
  assert(input.scalar_type() == at::ScalarType::Byte);

  uint8_t *input_d = input.data_ptr<uint8_t>();
  int n_element = input.size(0);

  Overflow op;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  // Determine temporary device storage requirements for exclusive
  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, input_d,
                                 op, (uint8_t)0, n_element);
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, input_d,
                                 op, (uint8_t)0, n_element);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(d_temp_storage);
  return;
}

__device__ uint32_t get_idx_in_little_endian(uint32_t bit_idx_in_big_endian) {
  return (3 - bit_idx_in_big_endian / 8) * 8 + bit_idx_in_big_endian % 8;
}

__device__ __host__ uint32_t *
get_bs_basic_by_id(uint32_t *bs_basic, uint32_t n_unit_basic, uint32_t id) {
  return bs_basic + n_unit_basic * id;
}

__device__ uint32_t swapEndian(uint32_t value) {
  return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) |
         ((value & 0x00FF0000) >> 8) | ((value & 0xFF000000) >> 24);
}

extern "C" __global__ void byte_stream_transpose_kernel(uint32_t *bytestream,
                                                        uint32_t *bitstream,
                                                        uint32_t n_unit) {
  uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t multi_input_id = blockIdx.y;
  uint32_t n_unit_basic = n_unit / 8;
  bytestream = bytestream + multi_input_id * n_unit;
  bitstream = bitstream + multi_input_id * n_unit;
  // 1 thread -> 8 units -> 8 uint32-> 8*32 bits
  uint32_t unit_id = thread_id * 8;
  while (unit_id < n_unit) {
    int pos_unit_in_basic = unit_id / 8;
    uint32_t bytestream_block[8];
    uint32_t bitstream_block[8];
#pragma unroll 8
    for (int i = 0; i < 8; i++) {
      if (unit_id + i < n_unit) {
        bytestream_block[i] = swapEndian(bytestream[unit_id + i]);
      }
      bitstream_block[i] = 0;
    }
#pragma unroll 8
    for (int i = 0; i < 8; i++) {
#pragma unroll 8
      for (int j = 0; j < 8; j++) {
        bitstream_block[i] |= (((bytestream_block[j]) >> (31 - i)) & 1)
                              << (31 - (4 * j + 0));
        bitstream_block[i] |= (((bytestream_block[j]) >> (31 - i - 8)) & 1)
                              << (31 - (4 * j + 1));
        bitstream_block[i] |= (((bytestream_block[j]) >> (31 - i - 16)) & 1)
                              << (31 - (4 * j + 2));
        bitstream_block[i] |= (((bytestream_block[j]) >> (31 - i - 24)) & 1)
                              << (31 - (4 * j + 3));
      }
      if (unit_id + i < n_unit) {
        bitstream[pos_unit_in_basic + i * n_unit_basic] = (bitstream_block[i]);
      }
    }
    unit_id += gridDim.x * blockDim.x * 8;
  }
}

extern "C" void byte_stream_transpose(at::Tensor bytestream,
                                      at::Tensor bitstream) {
  assert(bytestream.scalar_type() == at::ScalarType::Byte);
  assert(bitstream.scalar_type() == at::ScalarType::UInt32);
  assert(bytestream.dim() == 3);
  assert(bitstream.dim() == 4);
  uint32_t *bytestream_d =
      reinterpret_cast<uint32_t *>(bytestream.data_ptr<uint8_t>());
  uint32_t *bitstream_d = bitstream.data_ptr<uint32_t>();
  int split_input = bytestream.size(0);
  int multi_input = bytestream.size(1);
  int n_unit_byte = bytestream.size(2);
  int n_unit = n_unit_byte / 4; // uint32 per input
  assert(bitstream.size(2) == 8);
  assert(bitstream.size(3) == bytestream.size(2) / 32);
  assert(bytestream.size(2) % 8 == 0);
  int block_size = 512;
  dim3 grid_size((n_unit + block_size * 8 - 1) / (block_size * 8), multi_input,
                 1);
  byte_stream_transpose_kernel<<<grid_size, block_size>>>(bytestream_d,
                                                          bitstream_d, n_unit);
  CUDA_CHECK(cudaDeviceSynchronize());
  return;
}