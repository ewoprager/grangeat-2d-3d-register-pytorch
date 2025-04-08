import os
import time
import sys
import argparse
import logging.config
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("simpleExample")

import torch
import matplotlib.pyplot as plt

import Extension as ExtensionTest

from benchmark_radon2d import benchmark_radon2d, benchmark_dRadon2dDR
from benchmark_radon3d import benchmark_radon3d, benchmark_dRadon3dDR
from register import register
from benchmark_resample_sinogram3d import benchmark_resample_sinogram3d
from benchmark_similarity import benchmark_similarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-p", "--ct-path", type=str, help="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache", help="")
    parser.add_argument("-i", "--ignore-cache", action='store_true', help="")
    args = parser.parse_args()

    if not os.path.exists(args["cache_directory"]):
        os.makedirs(args["cache_directory"])

    # benchmark_radon2d("/home/eprager/Documents/Data/4th year project/First/x_ray/x_ray.dcm")
    # benchmark_dRadon2dDR("/home/eprager/Documents/Data/4th year project/First/x_ray/x_ray.dcm")
    # benchmark_radon3d(sys.argv[1])
    # benchmark_dRadon3dDR(sys.argv[1])
    register(args["ct_path"], cache_directory=args["cache_directory"], load_cached=not(args["ignore_cache"]),
             regenerate_drr=False)

    # register(None, cache_directory=cache_directory, load_cached=False, regenerate_drr=True, save_to_cache=False)  # benchmark_resample_sinogram3d(sys.argv[1], cache_directory=cache_directory, load_cached=True, save_to_cache=False)  # benchmark_similarity()
