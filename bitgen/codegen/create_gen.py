from .gen import Generator
from .gen_python import PythonGenerator
from .gen_cuda import CudaGenerator
from .gen_torch import TorchGenerator

def create_generator(type: str):
    if type == "bitstream":
        Gen = Generator
    elif type == "python":
        Gen = PythonGenerator
    elif type == "cuda":
        Gen = CudaGenerator
    elif type == "torch":
        Gen = TorchGenerator
    else:
        raise ValueError("Unknown lower type.")
    return Gen
