import os
import torch
import numpy as np
import glob

from bitgen.backend.cuda.cuda_kernel import CUDA_Kernel, CUDA_Args


def init_args(kernel, regex_func_num, bitstream: torch.Tensor):
    assert bitstream.dim() == 4
    cuda_args = CUDA_Args()

    n_unit_basic = bitstream.size(-1)
    n_char = bitstream.size(-1) * bitstream.element_size() * 8
    n_split_input = bitstream.size(0)
    n_multi_input = bitstream.size(1)
    print(
        "n_unit_basic:",
        n_unit_basic,
        " n_char:",
        n_char,
        " n_split_input:",
        n_split_input,
    )
    assert n_split_input == 1

    result_shape = (
        n_split_input,
        n_multi_input,
        regex_func_num,
        n_unit_basic,
    )
    tmp_streams_shape = (
        n_split_input,
        n_multi_input,
        tmp_streams_num,
        n_unit_basic,
    )
    bitstream_shape = bitstream.shape

    def cal_tensor_mem_mb(shape, dtype=torch.uint32):
        byte_num = np.prod(shape) * torch.tensor([], dtype=dtype).element_size()
        return byte_num / (1024**2)

    print(
        f"bitstream shape: {bitstream_shape}, Mem: {cal_tensor_mem_mb(bitstream_shape, torch.uint32)} MB"
    )
    print(
        f"result shape: {result_shape}, Mem: {cal_tensor_mem_mb(result_shape, torch.uint32)} MB"
    )
    print(
        f"tmp_streams shape: {tmp_streams_shape}, Mem: {cal_tensor_mem_mb(tmp_streams_shape, torch.uint32)} MB"
    )
    print(
        "Total Mem: ",
        cal_tensor_mem_mb(bitstream_shape, torch.uint32)
        + cal_tensor_mem_mb(result_shape, torch.uint32)
        + cal_tensor_mem_mb(tmp_streams_shape, torch.uint32),
        " MB",
    )

    result = torch.zeros(
        result_shape,
        dtype=torch.uint32,
    ).cuda()

    tmp_streams = torch.zeros(
        tmp_streams_shape,
        dtype=torch.uint32,
    ).cuda()

    result = result.contiguous()
    bitstream = bitstream.contiguous()
    tmp_streams = tmp_streams.contiguous()

    for input_split_id in range(n_split_input):
        # The following code example is not intuitive
        # Subject to change in a future releaseq
        bitstream_split_ptr = [
            int(
                bitstream.data_ptr()
                + input_split_id
                * bitstream.size(1)
                * bitstream.size(2)
                * bitstream.size(3)
                * bitstream.element_size()
            )
        ]
        result_ptr = [int(result.data_ptr())]
        tmp_streams_ptr = [int(tmp_streams.data_ptr())]
        np_n_unit_basic = [np.uint32(n_unit_basic)]
        np_n_char = [np.uint32(n_char)]

        d_input_stream_uint32 = np.array(bitstream_split_ptr, dtype=np.uint64)
        d_result = np.array(result_ptr, dtype=np.uint64)
        d_tmp_streams = np.array(tmp_streams_ptr, dtype=np.uint64)
        d_n_unit_basic = np.array(np_n_unit_basic, dtype=np.uint64)
        d_n_char = np.array(np_n_char, dtype=np.uint64)

        args = (
            d_input_stream_uint32,
            d_n_unit_basic,
            d_n_char,
            d_result,
            d_tmp_streams,
        )
        cuda_args.args.append(args)

        kernel_args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)
        cuda_args.kernel_args.append(kernel_args)

    return cuda_args, result, tmp_streams


# Don't pass tensor created by nvrtc.
def run_kernel(kernel, cuda_args_list, bitstream):
    # bitstream and result is not used in this function, just to unify the interface
    for cuda_args in cuda_args_list:
        kernel.run(cuda_args)
        # torch.cuda.synchronize()
    return None


def init_kernel(regex_func_num, input_stream_tensor, basic_stream_tensor=None, cfg={}):
    cu_kernel_files = glob.glob(os.path.join(os.path.dirname(__file__), "kernel*.cu"))
    for cu_file_id in range(len(cu_kernel_files)):
        cu_kernel_files[cu_file_id] = os.path.join(
            os.path.dirname(__file__), cu_kernel_files[cu_file_id]
        )
    kernel = CUDA_Kernel(
        code_path=cu_kernel_files,
        kernel_name="KernelGenerated",
        parallel_compile=cfg.get("parallel_compile_cuda"),
    )
    kernel.init()
    kernel.set_size(grid_size, block_size)
    kernel.kernel_info()
    (
        cuda_args,
        result,
        tmp_streams,
    ) = init_args(kernel, regex_func_num, basic_stream_tensor)
    return (
        result,
        tmp_streams,
        kernel,
        lambda bitstream: run_kernel(kernel, cuda_args.kernel_args, bitstream),
        None
    )
