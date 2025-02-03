import os
import time
import torch
import matplotlib.pyplot as plt
import sys

import Extension as ExtensionTest

import test_radon2d
import test_radon3d
from register import register

if __name__ == "__main__":
    cache_directory = "cache"
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)
    # test_radon2d.benchmark_radon2d("/home/eprager/Documents/Data/4th year project/First/x_ray/x_ray.dcm")
    # test_radon2d.benchmark_dRadon2dDR("/home/eprager/Documents/Data/4th year project/First/x_ray/x_ray.dcm")
    # test_radon3d.benchmark_radon3d(sys.argv[1])
    # test_radon3d.benchmark_dRadon3dDR(sys.argv[1])
    register(sys.argv[1], cache_directory=cache_directory, load_cached=True, regenerate_drr=True)
