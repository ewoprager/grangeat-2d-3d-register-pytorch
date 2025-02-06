import os.path
import sys
import glob
import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME

extension_name: str = "ExtensionTest"
cpp_version: str = "-std=c++17"
cpu_source_files: list[str] = glob.glob("src/*.cpp")
cuda_source_files: list[str] = glob.glob("src/*.cu")

debug: bool = ("--debug" in sys.argv)
if debug:
    sys.argv.remove("--debug")
    print("Building in debug mode")

no_cuda: bool = ("--no-cuda" in sys.argv)
if no_cuda:
    sys.argv.remove("--no-cuda")

use_cuda: bool = (no_cuda == False) and torch.cuda.is_available() and (CUDA_HOME is not None)
if use_cuda:
    print("Building with CUDA")
elif no_cuda:
    print("Building without CUDA")
else:
    print("CUDA not available, building without")

extension_object = CUDAExtension if use_cuda else CppExtension

source_files: list[str] = cpu_source_files + cuda_source_files if use_cuda else cpu_source_files

extra_compile_args = {
    "cxx": [
        cpp_version,
        "-O0" if debug else "-O3",
        "-fdiagnostics-color=always"
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
