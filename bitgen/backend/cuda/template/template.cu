
// #include <cub/block/block_scan.cuh> // Update cub version
// #include <cooperative_groups.h>
// namespace cg = cooperative_groups;
typedef unsigned int uint32_t;

// #define DEBUG_PRINT
#define DEBUG_PRINT_UNIT 0

#define UINT32_MAX (4294967295U)

// template <typename T> struct BinaryScanFunctor {
//   __device__ T operator()(const T &a, const T &b) {
//     // (a->carry & b->max) | (b->carry)
//     return (((a & (b >> 1)) | b) & 1);
//   }
// };

extern "C" {

static __device__ __forceinline__ void swap_pointer(uint32_t **a,
                                                    uint32_t **b) {
  uint32_t *temp = *a;
  *a = *b;
  *b = temp;
}

// static __device__ __forceinline__ uint32_t swapEndian(uint32_t value) {
//   return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) |
//          ((value & 0x00FF0000) >> 8) | ((value & 0xFF000000) >> 24);
// }

// static __device__ __forceinline__ void print_uint32_binary(uint32_t n) {
//   // n = swapEndian(n);
//   for (int i = 31; i >= 0; i--) {
//     printf("%d", (n >> i) & 1);
//     if (i % 8 == 0)
//       printf(" ");
//   }
//   printf("\n");
// }

// static __device__ __forceinline__ void
// print_uint32_binary_debug(const char *msg, uint32_t n) {
// #ifdef DEBUG_PRINT
//   if (threadIdx.x == 0 && blockIdx.x == 0) {
//     printf("%-32s", msg);
//     print_uint32_binary(n);
//   }
// #endif
// }

// static __device__ __forceinline__ void print_32char_debug(const char *msg,
//                                                           char *n) {
// #ifdef DEBUG_PRINT
//   if (threadIdx.x == 0 && blockIdx.x == 0) {
//     printf("%-32s", msg);
//     for (int i = 0; i < 32; i++) {
//       printf("%c", n[i]);
//       if (i % 8 == 7)
//         printf(" ");
//     }
//     printf("\n");
//   }
// #endif
// }

// static __device__ __forceinline__ uint32_t prefix_sum(uint32_t a, uint32_t b)
// {
//   // (a->carry & b->max) | (b->carry)
//   return (((a & (b >> 1)) | b) & 1);
// };

// struct BlockPrefixCallbackOp {
//   uint32_t running_total;
//   __device__ BlockPrefixCallbackOp(uint32_t running_total)
//       : running_total(running_total) {}
//   __device__ uint32_t operator()(uint32_t block_aggregate) {
//     uint32_t old_prefix = running_total;
//     running_total += block_aggregate;
//     return old_prefix;
//   }
// };

// static __device__ __forceinline__ void
// bs_stream_add(uint32_t *op1_stream, uint32_t *op2_stream, uint32_t
// *carry_max,
//               uint32_t *result_stream, uint32_t n_unit_basic) {
//   for (uint32_t i = 0; i < ceil(1.0 * n_unit_basic / (blockDim.x)); i++) {
//     int idx = i * (blockDim.x) + threadIdx.x;
//     uint32_t op1 = __brev(swapEndian(op1_stream[idx]));
//     uint32_t op2 = __brev(swapEndian(op2_stream[idx]));
//     uint32_t r = (op1) + (op2);
//     if (idx < n_unit_basic) {
//       if (r < op1)
//         carry_max[idx] = 1;
//       if (r == UINT32_MAX)
//         carry_max[idx] = 2;
//     }
//     result_stream[idx] = r;
//   }
//   __syncthreads();

//   // https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockScan.html#id8
//   using BlockScan = cub::BlockScan<uint32_t, BLOCK_SIZE>;
//   __shared__ typename BlockScan::TempStorage temp_storage;
//   BlockPrefixCallbackOp prefix_op(0);
//   BinaryScanFunctor<uint32_t> binary_op;
//   for (uint32_t i = 0; i < ceil(1.0 * n_unit_basic / (blockDim.x)); i++) {
//     int idx = i * (blockDim.x) + threadIdx.x;
//     uint32_t thread_data = carry_max[idx];
//     BlockScan(temp_storage)
//         .ExclusiveScan(thread_data, thread_data, binary_op, prefix_op);
//     __syncthreads();
//     carry_max[idx] = thread_data;
//     result_stream[idx] += swapEndian(__brev(result_stream[idx] +
//     thread_data));
//   }
//   __syncthreads();
// }

static __device__ __forceinline__ uint32_t bs_not(uint32_t bs_input) {
  uint32_t result = ~bs_input;
#ifdef DEBUG_PRINT
  print_uint32_binary_debug("    bs_not op1: ", bs_input);
  print_uint32_binary_debug("bs_not result: ", result);
#endif
  return result;
}

static __device__ __forceinline__ uint32_t bs_and(uint32_t op1, uint32_t op2) {
  uint32_t result = op1 & op2;
#ifdef DEBUG_PRINT
  print_uint32_binary_debug("    bs_and op1: ", op1);
  print_uint32_binary_debug("    bs_and op2: ", op2);
  print_uint32_binary_debug("bs_and result: ", result);
#endif
  return result;
}

static __device__ __forceinline__ uint32_t bs_or(uint32_t op1, uint32_t op2) {
  uint32_t result = op1 | op2;
#ifdef DEBUG_PRINT
  print_uint32_binary_debug("    bs_or op1: ", op1);
  print_uint32_binary_debug("    bs_or op2: ", op2);
  print_uint32_binary_debug("bs_or result: ", result);
#endif
  return result;
}

static __device__ __forceinline__ uint32_t bs_xor(uint32_t op1, uint32_t op2) {
  uint32_t result = op1 ^ op2;
#ifdef DEBUG_PRINT
  print_uint32_binary_debug("    bs_xor op1: ", op1);
  print_uint32_binary_debug("    bs_xor op2: ", op2);
  print_uint32_binary_debug("bs_xor result: ", result);
#endif
  return result;
}

// static __device__ __forceinline__ uint32_t bs_scan_thru(uint32_t op1,
//                                                         uint32_t op2) {
//   // TODO(tge): Implement this function
//   uint32_t result = op1;
//   return result;
// }

// static __device__ __forceinline__ void
// bs_scan_thru_stream(uint32_t *op1_stream, uint32_t *op2_stream,
//                     uint32_t *carry_stream, uint32_t *add_stream,
//                     uint32_t *result_stream, uint32_t n_unit_basic) {

//   bs_stream_add(op1_stream, op2_stream, carry_stream, add_stream,
//   n_unit_basic);
//   __syncthreads();
//   for (uint32_t i = 0; i < ceil(1.0 * n_unit_basic / (blockDim.x)); i += 1) {
//     int idx = i * (blockDim.x) + threadIdx.x;
//     uint32_t add = add_stream[idx];
//     uint32_t op2_not = bs_not(op2_stream[idx]);
//     uint32_t result = bs_and(add, op2_not);
//     if (idx < n_unit_basic) {
//       carry_stream[idx] = result;
//     }
//   }
//   __syncthreads();
// }

// static __device__ __forceinline__ uint32_t bs_match_star(uint32_t op1,
//                                                          uint32_t op2) {
//   // TODO(tge): Implement this function
//   uint32_t result = op1;
//   return result;
// }

static __device__ __forceinline__ uint32_t
get_value_with_bit_offset_right(uint32_t *advance_memory, int offset) {
  // if (unit_id >= advance_memory_size) {
  //   printf("ERROR: unit_id should be less than n_unit\n");
  //   return 0;
  // }
  int unit_offset = offset / 32;
  int bit_offset = offset % 32;
  if (threadIdx.x < unit_offset)
    return 0;
  uint32_t bs_value = advance_memory[threadIdx.x - unit_offset];
  if (threadIdx.x == unit_offset)
    return bs_value >> bit_offset;
  uint32_t bs_value_prev = advance_memory[threadIdx.x - unit_offset - 1];
  uint32_t bs_value_right =
      (bit_offset == 0) ? 0 : (bs_value_prev << (32 - bit_offset));
  bs_value = (bs_value >> bit_offset) | bs_value_right;
  return bs_value;
}

static __device__ __forceinline__ uint32_t
get_value_with_bit_offset_right2(uint32_t *advance_memory, int offset) {
  int unit_offset = offset >> 5;
  int bit_offset = offset & 31;
  int first_unit = unit_offset;

  uint32_t bs_value =
      threadIdx.x >= first_unit ? advance_memory[threadIdx.x - unit_offset] : 0;
  uint32_t bs_value_prev = threadIdx.x > first_unit
                               ? advance_memory[threadIdx.x - unit_offset - 1]
                               : 0;
  return __funnelshift_r(bs_value, bs_value_prev, bit_offset);
}

static __device__ __forceinline__ uint32_t BSAdvanceRightFunction(
    uint32_t bs_input, uint32_t n_bits, uint32_t *advance_memory) {
  // Write to anoterh chunk of memory, exchange data with other threads in the
  // same CTA return bs_input+1;
  // TODO(tge): Fix unit_id and advance_memory_size=
  uint32_t result = get_value_with_bit_offset_right(advance_memory, n_bits);
#ifdef DEBUG_PRINT
  print_uint32_binary_debug("    bs_advance op1: ", bs_input);
  print_uint32_binary_debug("bs_advance result: ", result);
#endif
  return result;
}
static __device__ __forceinline__ uint32_t BSAdvanceRightFunctionSync(
    uint32_t bs_input, uint32_t n_bits, uint32_t *advance_memory) {
  // Write to anoterh chunk of memory, exchange data with other threads in the
  // same CTA return bs_input+1;
  // TODO(tge): Fix unit_id and advance_memory_size
  __syncthreads();
  advance_memory[threadIdx.x] = bs_input;
  __syncthreads();
  uint32_t result = get_value_with_bit_offset_right(advance_memory, n_bits);
#ifdef DEBUG_PRINT
  print_uint32_binary_debug("    bs_advance op1: ", bs_input);
  print_uint32_binary_debug("bs_advance result: ", result);
#endif
  return result;
}

// static __device__ __forceinline__ uint32_t
// get_value_with_bit_offset_right_block(
//     uint32_t *advance_memory, const uint32_t advance_memory_size,
//     uint32_t unit_id, int offset, uint32_t *prev_block) {
//   // offset = unit_offset * 32 + bit_offset
//   int unit_offset = offset / 32;
//   int bit_offset = offset % 32;
//   int bound = (offset + 32) / 32;
//   uint32_t block_result_value = 0;
//   if (unit_id < bound) {
//     uint32_t prev_block_value =
//         prev_block[advance_memory_size - 1 - unit_offset];
//     uint32_t block_value = advance_memory[unit_id - unit_offset];
//     block_result_value =
//         (block_value >> bit_offset) | (prev_block_value << (32 -
//         bit_offset));
//   } else {
//     uint32_t block_value_prev = advance_memory[unit_id - unit_offset - 1];
//     uint32_t block_value = advance_memory[unit_id - unit_offset];
//     block_result_value =
//         (block_value >> bit_offset) | (block_value_prev << (32 -
//         bit_offset));
//   }
//   return block_result_value;
// }

// static __device__ __forceinline__ uint32_t
// BlockAdvanceRightFunction(uint32_t bs_input, uint32_t n_bits, uint32_t
// unit_id,
//                           uint32_t *advance_memory, uint32_t *prev_block) {
//   advance_memory[unit_id] = bs_input;
//   __syncthreads();
//   uint32_t result = get_value_with_bit_offset_right(
//       advance_memory, BLOCK_SIZE, unit_id, n_bits);
//   print_uint32_binary_debug("    bs_advance op1: ", bs_input);
//   print_uint32_binary_debug("bs_advance result: ", result);
//   return result;
// }

static __device__ __forceinline__ uint32_t
get_value_with_bit_offset_left(uint32_t *advance_memory, int offset) {
  int unit_offset = offset >> 5;
  int bit_offset = offset & 31;
  uint32_t last_unit = BLOCK_SIZE - unit_offset - 1;

  uint32_t bs_value =
      threadIdx.x <= last_unit ? advance_memory[threadIdx.x + unit_offset] : 0;
  uint32_t bs_value_next = threadIdx.x <= (last_unit - 1)
                               ? advance_memory[threadIdx.x + unit_offset + 1]
                               : 0;
  return __funnelshift_l(bs_value_next, bs_value, bit_offset);
}

// Only used for pass cc_advance
static __device__ __forceinline__ uint32_t BSAdvanceLeftFunction(
    uint32_t bs_input, uint32_t n_bits, uint32_t *advance_memory) {
  // Write to anoterh chunk of memory, exchange data with other threads in the
  // same CTA
  // advance_memory[unit_id] = bs_input;
  // __syncthreads();
  uint32_t result = get_value_with_bit_offset_left(advance_memory, n_bits);
#ifdef DEBUG_PRINT
  print_uint32_binary_debug("    bs_advance op1: ", bs_input);
  print_uint32_binary_debug("bs_advance result: ", result);
#endif
  return result;
}
static __device__ __forceinline__ uint32_t BSAdvanceLeftFunctionSync(
    uint32_t bs_input, uint32_t n_bits, uint32_t *advance_memory) {
  // Write to anoterh chunk of memory, exchange data with other threads in the
  // same CTA
  __syncthreads();
  advance_memory[threadIdx.x] = bs_input;
  __syncthreads();
  uint32_t result = get_value_with_bit_offset_left(advance_memory, n_bits);
#ifdef DEBUG_PRINT
  print_uint32_binary_debug("    bs_advance op1: ", bs_input);
  print_uint32_binary_debug("bs_advance result: ", result);
#endif
  return result;
}

static __device__ __noinline__ bool bs_all_zeros(uint32_t *bs,
                                                 uint32_t n_unit_basic) {
  __shared__ uint32_t result;
  result = 0;
  __syncthreads();
  for (uint32_t idx = threadIdx.x; idx < n_unit_basic; idx += blockDim.x) {
    if (bs[idx] != 0) {
      atomicOr(&result, 1);
    }
  }
  __syncthreads();
  if (result != 0) {
    return false;
  }
  return true;
}

static __device__ __forceinline__ bool block_all_zeros(uint32_t bs,
                                                       uint32_t *zero_flag) {
  __syncthreads();
  if (threadIdx.x == 0)
    *zero_flag = 0;
  __syncthreads();
  atomicOr(zero_flag, bs);
  __syncthreads();
  return *zero_flag == 0;

  // __shared__ uint32_t any_non_zero;
  // if (threadIdx.x == 0) any_non_zero = 0;
  // __syncthreads();
  // bool has_bits = (bs != 0);
  // bool warp_any = __any_sync(0xFFFFFFFF, has_bits);
  // if (warp_any && threadIdx.x % 32 == 0) atomicOr(&any_non_zero, 1);
  // __syncthreads();
  // return (any_non_zero == 0);
}

static __device__ __forceinline__ bool block_all_zeros2(uint32_t bs, uint32_t *zero_flag) {
  if (threadIdx.x == 0) {
    *zero_flag = 0;
  }
  __syncthreads();
  if (bs != 0) {
    *zero_flag = 1;
  }
  __syncthreads();
  return *zero_flag == 0;
}

// static __device__ __forceinline__ bool warp_all_zeros(uint32_t bs) {
//   return __all_sync(0xFFFFFFFF, bs == 0);
// }

// static __device__ __forceinline__ uint32_t bs_ones_num(uint32_t *bs,
//                                                        uint32_t n_unit_basic)
//                                                        {
//   __shared__ uint32_t result;
//   result = 0;
//   __syncthreads();
//   for (uint32_t i = 0; i < ceil(1.0 * n_unit_basic / (blockDim.x)); i += 1) {
//     int idx = i * (blockDim.x) + threadIdx.x;
//     if (idx < n_unit_basic && bs[idx] != 0) {
//       // printf("bs[%d]: %d\n", idx, bs[idx]);
//       atomicAdd(&result, __popc(bs[idx]));
//     }
//     __syncthreads();
//   }
//   return result;
// }

// static __device__ __forceinline__ bool match(uint32_t offset,
//                                              const uint32_t *symbol_set) {
//   int pos = (offset / 32);
//   return symbol_set[pos] & (1 << (offset % 32));
// }

// static __device__ __forceinline__ uint32_t
// bs_match(char *input, uint32_t length, const uint32_t *symbol_set) {
//   uint32_t result = 0;
// #pragma unroll 1
//   for (int i = 0; i < length; i++) {
//     char c = input[i];
//     if (match(c, symbol_set))
//       result |= (0x1 << (31 - i));
//   }
//   result = swapEndian(result);
//   print_32char_debug("bs_match input: ", input);
//   print_uint32_binary_debug("bs_match result: ", result);
//   return result;
// }

// static __device__ __forceinline__ char *
// get_char_data(uint32_t *input_stream, uint32_t unit_id, uint32_t offset) {
//   char *input_stream_char = reinterpret_cast<char *>(input_stream);
//   return input_stream_char + unit_id * 32 + offset;
// }

// static __device__ __forceinline__ void
// set_bitstream_data(uint32_t *output_stream, uint32_t unit_id,
//                    uint32_t bit_offset, uint32_t value,
//                    uint32_t n_unit_output) {

//   if (unit_id >= n_unit_output)
//     return;
//   if (bit_offset == 0) {
//     output_stream[unit_id] = value;
//   } else {
//     uint32_t mask = (0xffffffff >> bit_offset);
//     value = swapEndian(value);
//     uint32_t orig_value_1 = swapEndian(output_stream[unit_id]);
//     uint32_t value_1 =
//         swapEndian((orig_value_1 & ~mask) | (value >> bit_offset));
//     atomicOr(&output_stream[unit_id], value_1);
//     if (unit_id + 1 < n_unit_output) {
//       uint32_t orig_value_2 = swapEndian(output_stream[unit_id + 1]);
//       uint32_t value_2 = swapEndian(value << (32 - bit_offset));
//       atomicOr(&output_stream[unit_id + 1], value_2);
//     }
//   }
// }

// static __device__ __forceinline__ uint32_t
// get_idx_in_little_endian(uint32_t bit_idx_in_big_endian) {
//   return (3 - bit_idx_in_big_endian / 8) * 8 + bit_idx_in_big_endian % 8;
// }

// static __device__ __forceinline__ uint32_t *transposed_256b(char *b256) {
//   uint32_t *bs = reinterpret_cast<uint32_t *>(b256);
//   uint32_t *bs_basic = (uint32_t *)malloc(8 * sizeof(uint32_t));
//   for (int i = 0; i < 8; i++) {
//     uint32_t block_data = bs[i];
//     for (int j = 0; j < 32; j++) {
//       int real_j = get_idx_in_little_endian(j);
//       int bs_basic_id = j % 8;
//       int pos_bit_in_basic = 31 - (i * 4 + (j / 8));
//       pos_bit_in_basic = get_idx_in_little_endian(pos_bit_in_basic);
//       uint32_t *bs_basic_i = bs_basic + bs_basic_id;
//       bs_basic_i[0] |= ((block_data >> (31 - real_j)) & 1) <<
//       pos_bit_in_basic;
//     }
//   }
//   return bs_basic;
// }
}
