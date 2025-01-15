import sys
import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME

extension_name: str = "ExtensionTest"
cpp_version: str = "-std=c++17"
cpu_source_files: list[str] = ["radon.cpp"]
cuda_source_files: list[str] = ["radonCUDA.cu"]

debug: bool = "--debug" in sys.argv
if debug:
    sys.argv.remove("--debug")
    print("Building in debug mode")

use_cuda: bool = torch.cuda.is_available() and (CUDA_HOME is not None)
if use_cuda:
    print("Building with CUDA")
else:
    print("CUDA not available, building without")

extension_object = CUDAExtension if use_cuda else CppExtension

source_files: list[str] = cpu_source_files + cuda_source_files if use_cuda else cpu_source_files

extra_compile_args = {
    "cxx": [
        cpp_version,
        "-O0" if debug else "-O3",
        "-fdiagnostics-color=always",
    ],
    "nvcc": [
        "-O0" if debug else "-O3",
    ],
}
extra_link_args = []
if debug:
    extra_compile_args["cxx"].append("-g")
    extra_compile_args["nvcc"].extend(["-g", "-G", "-DTORCH_USE_CUDA_DSA=1", "-DCUDA_LAUNCH_BLOCKING=1"])
    extra_link_args.extend(["-O0", "-g"])

setup(name=extension_name,
      ext_modules=[extension_object(name=extension_name,
                                    sources=source_files,
                                    extra_compile_args=extra_compile_args,
                                    extra_link_args=extra_link_args)],
      cmdclass={'build_ext': BuildExtension})
