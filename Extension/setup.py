import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="ExtensionTest",
      ext_modules=[cpp_extension.CUDAExtension(name="ExtensionTest",
                                               sources=["mymuladd.cpp", "muladd.cu"],
                                               extra_compile_args=["-std=c++17"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
