import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="ExtensionTest",
      ext_modules=[cpp_extension.CppExtension(name="ExtensionTest",
                                              sources=["mymuladd.cpp"],
                                              extra_compile_args=["-std=c++20"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})