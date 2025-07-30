from cuda.bindings import driver, nvrtc, nvjitlink
from cuda.bindings.driver import (
    cuOccupancyMaxActiveBlocksPerMultiprocessor,
    cuFuncGetAttribute,
    CUfunction_attribute,
    cuFuncSetAttribute,
)
from cuda.bindings.runtime import (
    cudaGetDeviceProperties,
    cudaKernelSetAttributeForDevice,
    cudaFuncAttribute,
)
import os
import numpy as np
import torch
import time
import concurrent.futures
from tqdm import tqdm

from ...log import MyLogger
from ...tool.command_tool import exe_command
from ... import config as cfg


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                result[0].value, _cudaGetErrorEnum(result[0])
            )
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


class CUDA_Kernel:
    def __init__(self, code_path: str, kernel_name: str, parallel_compile=False):
        self.code_path = code_path
        self.code = None
        self.cuDevice = None
        self.module = None
        self.context = None
        self.stream = None
        self.kernel_name = kernel_name
        self.kernel = None
        self.grid_size = None
        self.block_size = None
        self.parallel_compile = parallel_compile

    def init_single(self):
        # Initialize CUDA Driver API
        checkCudaErrors(driver.cuInit(0))
        # Retrieve handle for device 0
        self.cuDevice = checkCudaErrors(driver.cuDeviceGet(0))
        # Derive target architecture for device 0
        major = checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.cuDevice,
            )
        )
        minor = checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.cuDevice,
            )
        )
        arch_arg = bytes(f"--gpu-architecture=sm_{major}{minor}", "ascii")
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home is None:
            raise ValueError("CUDA_HOME or CUDA_PATH must be set")
        cuda_include = bytes(
            "--include-path=" + os.path.join(cuda_home, "include"), "ascii"
        )
        conda_home = os.environ.get("CONDA_PREFIX")
        if conda_home is None:
            raise ValueError("CONDA_PREFIX must be set")
        cccl_include = bytes(
            "--include-path=" + os.path.join(conda_home, "include"),
            "ascii",
        )
        # Supported Compile Options in nvrtc: https://docs.nvidia.com/cuda/nvrtc/#supported-compile-options
        opts = [
            arch_arg,
            b"--dopt=on",
            b"--fmad=false",
            b"-std=c++17",
            b"--use_fast_math",
            cccl_include,
            cuda_include,
            b"--include-path=/usr/include",
            b"--ptxas-options=-v",
            b"--time=-",
            b"--split-compile=16",
            # b"--minimal",
            # b"--relocatable-device-code=true",
            # b"-G",
            # b"-lineinfo",
        ]
        MyLogger.debug(f"[Compilation] Compilation options: {opts}")

        # Create program
        headers = []
        includeNames = []
        numHeaders = len(headers)

        assert isinstance(self.code_path, list) and len(self.code_path) == 1
        with open(self.code_path[0], "r") as f:
            self.code = f.read()
        prog = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(
                str.encode(self.code), b"code.cu", numHeaders, headers, includeNames
            )
        )

        # Compile program
        MyLogger.info("[Compilation] Compiling CUDA code...")
        time_start = time.time()
        try:
            checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except Exception as e:
            n = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            MyLogger.error(
                f"[Compilation] nvcc output: {log.decode()}\n",
            )
            raise e
        finally:
            n = checkCudaErrors(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * n
            nvrtc.nvrtcGetProgramLog(prog, log)
            MyLogger.debug(
                f"[Compilation] nvcc output: {log.decode()}\n",
            )
        time_end = time.time()
        MyLogger.info(f"[Compilation] Compilation time: {time_end - time_start:.2f}s")

        # Get PTX from compilation
        ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
        ptx = b" " * ptxSize
        MyLogger.debug(f"[Compilation] PTX size: {ptxSize} bytes")
        checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))
        # Create context
        time_start = time.time()
        self.context = checkCudaErrors(driver.cuCtxCreate(0, self.cuDevice))
        # Load PTX as module data and retrieve function
        ptx = np.char.array(ptx)
        # Note: Incompatible --gpu-architecture would be detected here
        self.module = checkCudaErrors(driver.cuModuleLoadData(ptx.ctypes.data))
        self.kernel = checkCudaErrors(
            driver.cuModuleGetFunction(self.module, self.kernel_name.encode("utf-8"))
        )
        time_end = time.time()
        MyLogger.info(
            f"[Compilation] Module loading time: {time_end - time_start:.2f}s"
        )

    def parallel_compile_cmd(self, cmds):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            results = []
            check = True
            shell = False
            verbose = True
            futures = {
                executor.submit(exe_command, cmd, check, shell, verbose): cmd_id
                for cmd_id, cmd in enumerate(cmds)
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Compiling",
            ):
                cmd_id = futures[future]
                result = future.result()
                if result:
                    results.append((cmd_id, result))
                    # print("!!!!!!", result.stderr)
        return results

    def gen_nvcc_cubin_cmds(self, code_path, compute_capability="86", use_lto=False):
        cmds = []
        cubin_paths = []
        if use_lto:
            compute_capability = f"lto_{compute_capability}"
            output_type = "fatbin"
        else:
            compute_capability = f"sm_{compute_capability}"
            output_type = "cubin"
        register_num = cfg.get_config("cuda_register_num")
        cmd_template = f"nvcc -std=c++17 -O3 -use_fast_math --ptxas-options=-v -Xptxas -O3 -rdc=true -{output_type} -arch={compute_capability}"  # --ptxas-options=--register-usage-level=10
        if not use_lto:
            cmd_template += f" --maxrregcount={register_num}"
        for i, code_path in enumerate(code_path):
            root, _ = os.path.splitext(os.path.basename(code_path))
            cubin_path = os.path.join(
                os.path.dirname(code_path), "cubin", f"{root}.{output_type}"
            )
            os.makedirs(os.path.dirname(cubin_path), exist_ok=True)
            cmd = f"{cmd_template} -o {cubin_path} {code_path}"
            cmds.append(cmd)
            cubin_paths.append(cubin_path)
        return cmds, cubin_paths

    def init_parallel(self):
        if self.parallel_compile:
            assert isinstance(self.code_path, list) and len(self.code_path) >= 2

        # Initialize CUDA Driver API
        checkCudaErrors(driver.cuInit(0))
        # Retrieve handle for device 0
        self.cuDevice = checkCudaErrors(driver.cuDeviceGet(0))
        # Derive target architecture for device 0
        major = checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.cuDevice,
            )
        )
        minor = checkCudaErrors(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.cuDevice,
            )
        )
        use_lto = cfg.get_config("parallel_compile_cuda_lto")
        cmds, cubin_paths = self.gen_nvcc_cubin_cmds(
            self.code_path, compute_capability=f"{major}{minor}", use_lto=use_lto
        )

        MyLogger.info(f"[Compilation] Compiling CUDA code...")
        time_start = time.time()
        results = self.parallel_compile_cmd(cmds)
        time_end = time.time()
        MyLogger.info(f"[Compilation] Compilation time: {time_end - time_start:.2f}s")

        arch_arg = f"-arch=sm_{major}{minor}"
        if use_lto:
            lopts = [arch_arg, "-O3", "-dlto"]
            nvcc_output_type = nvjitlink.InputType.FATBIN
        else:
            lopts = [arch_arg, "-O3"]
            nvcc_output_type = nvjitlink.InputType.CUBIN

        handle = nvjitlink.create(len(lopts), lopts)

        MyLogger.info(f"[Compilation] Loading cubin files...")
        time_start = time.time()
        for cubin_path in cubin_paths:
            nvjitlink.add_file(handle, nvcc_output_type, cubin_path)
            # print(f"Loading {cubin_path}")
        nvjitlink.complete(handle)
        time_end = time.time()

        cubinSize = nvjitlink.get_linked_cubin_size(handle)
        print("cubinSize:", cubinSize)
        cubin = bytearray(cubinSize)
        nvjitlink.get_linked_cubin(handle, cubin)
        # with open("linked.cubin", "wb") as f:
        #     f.write(cubin)
        nvjitlink.destroy(handle)
        self.module = checkCudaErrors(driver.cuModuleLoadData(cubin))
        kernel_name = "KernelGenerated"
        self.kernel = checkCudaErrors(
            driver.cuModuleGetFunction(self.module, kernel_name.encode("utf-8"))
        )
        MyLogger.info(f"[Compilation] cubin loading time: {time_end - time_start:.2f}s")

    def init(self):
        if self.parallel_compile:
            self.init_parallel()
        else:
            self.init_single()

    def set_size(self, grid_size, block_size):
        assert len(grid_size) == 3
        assert len(block_size) == 3
        self.grid_size = grid_size
        self.block_size = block_size

    def kernel_info(self):
        blocksize = self.block_size[0] * self.block_size[1] * self.block_size[2]
        gridsize = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        maxActiveBlocks = checkCudaErrors(
            cuOccupancyMaxActiveBlocksPerMultiprocessor(
                func=self.kernel,
                blockSize=blocksize,
                dynamicSMemSize=0,
            )
        )
        # Information about the compute-device.
        # https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/runtime.html#cuda.bindings.runtime.cudaGetDeviceProperties
        cudaDeviceProp = checkCudaErrors(cudaGetDeviceProperties(self.cuDevice))
        print("CUDA device info:")
        print(f"    Device ame: {cudaDeviceProp.name}")
        print(f"    Clock rate (MHz): {torch.cuda.clock_rate()}")
        print(f"    Max Clock rate (MHz): {cudaDeviceProp.clockRate/1000}")
        print(f"    Memory clock rate (MHz): {cudaDeviceProp.memoryClockRate/1000}")
        print(f"    Multiprocessor count: {cudaDeviceProp.multiProcessorCount}")
        print(f"    Total global memory: {cudaDeviceProp.totalGlobalMem}")
        print(
            f"    Max shared memory per multiprocessor: {cudaDeviceProp.sharedMemPerMultiprocessor}"
        )
        print(f"    Shared memory per block: {cudaDeviceProp.sharedMemPerBlock}")
        print(f"    L2 cache size: {cudaDeviceProp.l2CacheSize}")
        # print(f"    Registers per block: {cudaDeviceProp.regsPerBlock}")
        # print(f"    Warp size: {cudaDeviceProp.warpSize}")
        # print(f"    Max threads per block: {cudaDeviceProp.maxThreadsPerBlock}")
        # print(f"    Max threads per multiprocessor: {cudaDeviceProp.maxThreadsPerMultiProcessor}")
        # print(f"    Max threads dim: {cudaDeviceProp.maxThreadsDim}")
        # print(f"    Max grid size: {cudaDeviceProp.maxGridSize}")
        print(
            f"    Max threads per multiprocessor: {cudaDeviceProp.maxThreadsPerMultiProcessor}"
        )
        print(f"    Compute capability: {cudaDeviceProp.major}.{cudaDeviceProp.minor}")

        # Information about a function
        # https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html#cuda.bindings.driver.cuFuncGetAttribute
        func_max_thread_per_block = checkCudaErrors(
            cuFuncGetAttribute(CUfunction_attribute(0), self.kernel)
        )
        func_shared_size_bytes = checkCudaErrors(
            cuFuncGetAttribute(CUfunction_attribute(1), self.kernel)
        )
        func_const_size_bytes = checkCudaErrors(
            cuFuncGetAttribute(CUfunction_attribute(2), self.kernel)
        )
        func_local_size_bytes = checkCudaErrors(
            cuFuncGetAttribute(CUfunction_attribute(3), self.kernel)
        )
        func_num_regs = checkCudaErrors(
            cuFuncGetAttribute(CUfunction_attribute(4), self.kernel)
        )
        func_ptx_version = checkCudaErrors(
            cuFuncGetAttribute(CUfunction_attribute(5), self.kernel)
        )
        func_max_dynamic_shared_size_bytes = checkCudaErrors(
            cuFuncGetAttribute(CUfunction_attribute(8), self.kernel)
        )
        func_occupancy = (maxActiveBlocks * blocksize / cudaDeviceProp.warpSize) / (
            cudaDeviceProp.maxThreadsPerMultiProcessor / cudaDeviceProp.warpSize
        )
        print(f"CUDA kernel info:")
        print(f"    Grid size: {self.grid_size}")
        print(f"    Block size: {self.block_size}")
        print(f"    PTX version: {func_ptx_version}")
        print(f"    Number of registers: {func_num_regs}")
        # print(f"    Max threads per block: {func_max_thread_per_block}")
        # print(f"    Max dynamic shared memory size (bytes): {func_max_dynamic_shared_size_bytes}")
        print(f"    Statically-allocated Shared memory size (bytes): {func_shared_size_bytes}")
        # print(f"    Constant memory size (bytes): {func_const_size_bytes}")
        # print(f"    Local memory size (bytes): {func_local_size_bytes}")
        print(
            f"    Theoretical cccupancy: {func_occupancy}",
            f"(blocksize={blocksize}, maxActiveBlocks={maxActiveBlocks}, maxThreadsPerMultiProcessor={cudaDeviceProp.maxThreadsPerMultiProcessor})",
        )
        print(
            f"    (to achieve theoretical occupancy, gridsize should >= {maxActiveBlocks}*{cudaDeviceProp.multiProcessorCount}={maxActiveBlocks * cudaDeviceProp.multiProcessorCount}, now gridsize={gridsize})"
        )
        if func_shared_size_bytes > cudaDeviceProp.sharedMemPerBlock:
            print(
                f"    Warning: shared memory size ({func_shared_size_bytes}) exceeds the limit of {cudaDeviceProp.sharedMemPerBlock} bytes"
            )
            cuFuncSetAttribute(
                self.kernel,
                CUfunction_attribute(8),  # cudaFuncAttributeMaxDynamicSharedMemorySize
                0,
            )
            cuFuncSetAttribute(
                self.kernel,
                CUfunction_attribute(
                    9
                ),  # cudaFuncAttributePreferredSharedMemoryCarveout
                0,
            )
            print("   Warning: Reset MaxDynamicSharedMemory")
            # cudaKernelSetAttributeForDevice(
            #     self.kernel,
            #     cudaFuncAttribute(8),  # cudaFuncAttributeMaxDynamicSharedMemorySize
            #     0,
            #     self.cuDevice
            # )
        self.stream = checkCudaErrors(driver.cuStreamCreate(0))  # CU_STREAM_DEFAULT = 0, CU_STREAM_NON_BLOCKING = 1

    def free(self):
        if self.stream:
            checkCudaErrors(driver.cuStreamDestroy(self.stream))
        if self.module:
            checkCudaErrors(driver.cuModuleUnload(self.module))
        if self.context:
            checkCudaErrors(driver.cuCtxDestroy(self.context))

    # def __del__(self):
    #     self.free()

    def run(self, cuda_args):
        checkCudaErrors(
            driver.cuLaunchKernel(
                self.kernel,
                self.grid_size[0],  # grid x dim
                self.grid_size[1],  # grid y dim
                self.grid_size[2],  # grid z dim
                self.block_size[0],  # block x dim
                self.block_size[1],  # block y dim
                self.block_size[2],  # block z dim
                0,  # dynamic shared memory
                self.stream,  # stream
                cuda_args,  # kernel arguments
                0,  # extra (ignore)
            )
        )
        checkCudaErrors(driver.cuStreamSynchronize(self.stream))


class CUDA_Args:
    def __init__(self):
        self.args = []  # Refer to the variable in cuda_args to not release the memory
        self.kernel_args = []
