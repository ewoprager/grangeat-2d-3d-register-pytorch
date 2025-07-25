import os
import sys
import glob
from typing import Tuple

import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME

extension_name: str = "reg23"
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

source_files: list[str] = (cpu_source_files + cuda_source_files) if use_cuda else cpu_source_files

macros: list[Tuple] = []
if use_cuda:
    macros.append(("USE_CUDA", None))

extra_compile_args = {
    "cxx": [
        cpp_version, "-O0" if debug else "-O3", "-fdiagnostics-color=always", "-Wno-c++20-extensions"
        # suppress C++20 extension warnings
    ], "nvcc": [
        "-O0" if debug else "-O3", "-diag-suppress=3288"  # suppress C++20 extension warnings
    ], }
extra_link_args = []
if debug:
    extra_compile_args["cxx"].extend(["-g"])
    extra_compile_args["nvcc"].extend(["-g", "-G", "-DTORCH_USE_CUDA_DSA=1", "-DCUDA_LAUNCH_BLOCKING=1"])
    extra_link_args.extend(["-O0", "-g"])

if torch.cuda.is_available():
    # Only compile for the compute capability of device 0:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "{}.{}".format(
        torch.cuda.get_device_capability(0)[0], torch.cuda.get_device_capability(0)[1])

setup(
    name=extension_name, ext_modules=[
        extension_object(
            name=extension_name, sources=source_files, define_macros=macros, extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)], cmdclass={'build_ext': BuildExtension})
