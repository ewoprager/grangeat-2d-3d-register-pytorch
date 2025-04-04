import os
import time
import torch
import matplotlib.pyplot as plt
import sys

import Extension as ExtensionTest

from benchmark_radon2d import benchmark_radon2d, benchmark_dRadon2dDR
from benchmark_radon3d import benchmark_radon3d, benchmark_dRadon3dDR
from register import register
from benchmark_resample_sinogram3d import benchmark_resample_sinogram3d
from benchmark_similarity import benchmark_similarity

if __name__ == "__main__":
    cache_directory = "cache"
    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory)

    # benchmark_radon2d("/home/eprager/Documents/Data/4th year project/First/x_ray/x_ray.dcm")
    # benchmark_dRadon2dDR("/home/eprager/Documents/Data/4th year project/First/x_ray/x_ray.dcm")
    # benchmark_radon3d(sys.argv[1])
    # benchmark_dRadon3dDR(sys.argv[1])
    register(sys.argv[1], cache_directory=cache_directory, load_cached=True, regenerate_drr=False)
    # register(None, cache_directory=cache_directory, load_cached=False, regenerate_drr=True, save_to_cache=False)
    # benchmark_resample_sinogram3d(sys.argv[1], cache_directory=cache_directory, load_cached=True, save_to_cache=False)
    # benchmark_similarity()
