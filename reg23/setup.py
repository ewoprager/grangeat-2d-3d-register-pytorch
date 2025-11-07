import os
import sys
import glob
from typing import Tuple
import subprocess

import pathlib
import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME

this_directory = pathlib.Path(__file__).parent
extension_name: str = "reg23"
cpp_version: str = "-std=c++17"
cpu_source_files: list[str] = glob.glob("src/*.cpp") + glob.glob("src/cpu/*.cpp")
cuda_source_files: list[str] = glob.glob("src/cuda/*.cu")
mps_source_files: list[str] = glob.glob("src/mps/*.mm")


def compile_metal_shaders(shader_dir: pathlib.Path, output_dir: pathlib.Path) -> pathlib.Path:
    """
    Requires the xcodebuild MetalToolChain to be installed.
    :param shader_dir:
    :param output_dir:
    :return:
    """
    # compile each .metal shader into a .air file
    output_dir.mkdir(exist_ok=True, parents=True)
    air_files = []
    for source in shader_dir.iterdir():
        if source.suffix != ".metal":
            continue
        air = output_dir / source.with_suffix(".air").name
        print("Compiling Metal shader: {}".format(str(source)))
        subprocess.check_call(["xcrun", "-sdk", "macosx", "metal", "-c", str(source), "-o", str(air)])
        air_files.append(str(air))

    # compile air files into a single metallib library
    metallib = output_dir / "default.metallib"
    print("Linking metallib...")
    subprocess.check_call(["xcrun", "-sdk", "macosx", "metallib", *air_files, "-o", str(metallib)])

    # clean up air files
    for air_file in air_files:
        pathlib.Path(air_file).unlink()

    # convert the library into a C-array
    header = output_dir / "default_metallib.h"
    header.write_text(subprocess.run(["xxd", "-i", str(metallib)], capture_output=True, text=True, check=True).stdout)

    # cleaning up library
    metallib.unlink()

    return metallib


class BuildExtensionWithMetal(BuildExtension):
    def run(self):
        shader_dir = pathlib.Path("src") / "mps" / "shaders"
        output_dir = pathlib.Path("include") / "reg23_mps"
        if shader_dir.is_dir():
            compile_metal_shaders(shader_dir, output_dir)
        else:
            print("Warning: No shader directory '{}'.".format(str(shader_dir)))
        super().run()


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

use_mps: bool = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if use_mps:
    print("Building with MPS")
else:
    print("MPS not available, building without")

extension_object = CUDAExtension if use_cuda else CppExtension

source_files: list[str] = cpu_source_files
if use_cuda:
    source_files.extend(cuda_source_files)
if use_mps:
    source_files.extend(mps_source_files)

include_dirs: list[str] = [str(this_directory / "include")]

macros: list[Tuple] = []
if use_cuda:
    macros.append(("USE_CUDA", None))

extra_compile_args = {  #
    "cxx": [  #
        cpp_version,  #
        "-O0" if debug else "-O3",  #
        "-fdiagnostics-color=always",  #
        "-Wno-c++20-extensions"  # suppress C++20 extension warnings

    ],  #
    "nvcc": [  #
        "-O0" if debug else "-O3",  #
        "-diag-suppress=3288"  # suppress C++20 extension warnings
    ],  #
}

if use_mps:
    # objc compiler support
    from distutils.unixccompiler import UnixCCompiler

    if '.mm' not in UnixCCompiler.src_extensions:
        UnixCCompiler.src_extensions.append('.mm')
        UnixCCompiler.language_map['.mm'] = 'objc'

    extra_compile_args['cxx'].extend(['-framework', 'Metal', '-framework', 'Foundation', '-ObjC++'])

extra_link_args = []
if debug:
    extra_compile_args["cxx"].extend(["-g"])
    extra_compile_args["nvcc"].extend(["-g", "-G", "-DTORCH_USE_CUDA_DSA=1", "-DCUDA_LAUNCH_BLOCKING=1"])
    extra_link_args.extend(["-O0", "-g"])
    macros.append(("DEBUG", None))

if use_cuda:
    # Only compile for the compute capability of device 0:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "{}.{}".format(torch.cuda.get_device_capability(0)[0],
                                                        torch.cuda.get_device_capability(0)[1])

setup(name=extension_name,  #
      ext_modules=[  #
          extension_object(name=extension_name,  #
                           sources=source_files,  #
                           include_dirs=include_dirs,  #
                           define_macros=macros,  #
                           extra_compile_args=extra_compile_args,  #
                           extra_link_args=extra_link_args)  #
      ],  #
      cmdclass={'build_ext': BuildExtensionWithMetal if use_mps else BuildExtension})
