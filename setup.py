import glob
import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from cuda.bindings import driver, nvrtc

root = Path(__file__).parent.resolve()


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


def get_sources():
    cpp_files = list(root.glob("bitgen/template/cuda/*.cpp"))
    cuda_files = list(root.glob("bitgen/template/cuda/*.cu"))
    sources = cpp_files + cuda_files
    # Optional: Remove the print statement or replace it with logging if needed
    print("Source files:", sources)
    return [str(file) for file in sources]


def get_cuda_arch_flags():
    # Initialize CUDA Driver API
    checkCudaErrors(driver.cuInit(0))
    # Retrieve handle for device 0
    cuDevice = checkCudaErrors(driver.cuDeviceGet(0))
    # Derive target architecture for device 0
    major = checkCudaErrors(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            cuDevice,
        )
    )
    minor = checkCudaErrors(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            cuDevice,
        )
    )
    arch_flag = f"-arch=sm_{major}{minor}"
    return [arch_flag]


setup(
    name="bitgen",
    version="0.1",
    packages=find_packages(),
    package_data={
        "bitgen": ["backend/cuda/template/*.cu", "datasets_small/*"],
    },
    # include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="bitgen.cuda",
             sources=[
                "bitgen/template/cuda/pybind.cpp",
                "bitgen/template/cuda/kernel.cu",
            ],
            include_dirs=["bitgen/template/cuda"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"] + get_cuda_arch_flags(),
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
