import time
import torch
import matplotlib.pyplot as plt
import sys

import Extension as ExtensionTest

import test_radon2d
import test_radon3d

if __name__ == "__main__":
    # test_radon2d.benchmark_radon2d()
    test_radon3d.benchmark_radon3d(sys.argv[1])
