import argparse
import gc
import os
import time
import pickle
import math
from typing import Type

import pathlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from notification import logs_setup
from registration.lib import geometry
from registration import data
from registration.lib import sinogram
from registration import pre_computed
from registration.lib.structs import *
from registration.lib import grangeat
from registration import plot_data

import reg23


def run_benchmark(*, cache_directory: str, ct_path: str | pathlib.Path, load_cached: bool = False,
                  save_to_cache: bool = True, downsample_factor: int = 1, plot: bool = False):
    # Load volumes with a number of downsample factors, and sinograms with a number of sizes
    volume, volume_spacing = data.load_volume(pathlib.Path(ct_path), downsample_factor=downsample_factor)
    volume = volume.to(dtype=torch.float32)

    scene_geometry = SceneGeometry(source_distance=1000.0)

    # Measure evaluation time of DRR
    repeats: int = 1
    assert repeats >= 1
    transformations = [Transformation.random_uniform() for _ in range(repeats)]
    p_matrix = SceneGeometry.projection_matrix(source_position=scene_geometry.source_position())
    ph_matrices = [torch.matmul(p_matrix, transformation.get_h()).to(dtype=torch.float32) for transformation in
                   transformations]
    h_matrix_invs = [transformation.inverse().get_h() for transformation in transformations]
    output_width = 1000
    output_height = 1000
    detector_spacing = torch.tensor([0.2, 0.2])

    # -----
    # DRR
    # -----
    logger.info("Projecting DRRs on CPU...")
    # CPU
    tic = time.time()
    for i in tqdm(range(repeats)):
        drr = reg23.project_drr(volume, volume_spacing, h_matrix_invs[i], scene_geometry.source_distance, output_width,
                                output_height, torch.zeros(2, dtype=torch.float64), detector_spacing)
    toc = time.time()
    drr_evaluation_time: float = (toc - tic) / float(repeats)
    logger.info("DRR projected on CPU; took {:.4f}s".format(drr_evaluation_time))
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(drr.cpu().numpy())
        axes.set_title("CPU")
        plt.colorbar(mesh)

    # CUDA
    if torch.cuda.is_available():
        logger.info("Projecting DRRs on CUDA...")
        device_cuda = torch.device('cuda')
        volume_cuda = volume.to(device=device_cuda)
        torch.cuda.synchronize()
        tic = time.time()
        for i in range(repeats):
            drr = reg23.project_drr(volume_cuda, volume_spacing, h_matrix_invs[i], scene_geometry.source_distance,
                                    output_width, output_height, torch.zeros(2, dtype=torch.float64), detector_spacing)
        torch.cuda.synchronize()
        toc = time.time()
        drr_evaluation_time: float = (toc - tic) / float(repeats)
        logger.info("DRR projected on CUDA; took {:.4f}s".format(drr_evaluation_time))
        if plot:
            _, axes = plt.subplots()
            mesh = axes.pcolormesh(drr.cpu().numpy())
            axes.set_title("CUDA")
            plt.colorbar(mesh)

    if False:
        logger.info("Projecting DRRs in python...")
        torch.cuda.synchronize()
        tic = time.time()
        for i in range(repeats):
            drr = geometry.generate_drr_python(volume, voxel_spacing=volume_spacing.to(device=volume.device),
                                               transformation=transformations[i].to(device=volume.device),
                                               scene_geometry=scene_geometry,
                                               output_size=torch.Size([output_height, output_width]),
                                               detector_spacing=detector_spacing)
        torch.cuda.synchronize()
        toc = time.time()
        drr_evaluation_time: float = (toc - tic) / float(repeats)
        logger.info("DRR projected in python; took {:.4f}s".format(drr_evaluation_time))
        if plot:
            _, axes = plt.subplots()
            mesh = axes.pcolormesh(drr.cpu().numpy())
            axes.set_title("DRR Python")
            plt.colorbar(mesh)

    # MPS
    if torch.mps.is_available():
        logger.info("Projecting DRRs on MPS...")
        device_mps = torch.device('mps')
        volume_mps = volume.to(device=device_mps)
        torch.mps.synchronize()
        tic = time.time()
        for i in range(repeats):
            drr = reg23.project_drr(volume_mps, volume_spacing, h_matrix_invs[i], scene_geometry.source_distance,
                                    output_width, output_height, torch.zeros(2, dtype=torch.float64), detector_spacing)
        torch.mps.synchronize()
        toc = time.time()
        drr_evaluation_time: float = (toc - tic) / float(repeats)
        logger.info("DRR projected on MPS; took {:.4f}s".format(drr_evaluation_time))
        if plot:
            _, axes = plt.subplots()
            mesh = axes.pcolormesh(drr.cpu().numpy())
            axes.set_title("MPS")
            plt.colorbar(mesh)

    plt.show()

CT_PATHS = ["/Users/eprager/Library/CloudStorage/OneDrive-UniversityofCambridge/CUED/4th Year Project/Data/First/ct"]
# CT_PATHS = ["/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct001",
#             "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct002",
#             "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct003",
#             "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct005",
#             "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct007",
#             "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct008",
#             "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct010",
#             "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct014"]
# XRAY_DICOM_PATHS = ["/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray001.dcm",
#                     "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray002.dcm",
#                     "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray003.dcm",
#                     "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray005.dcm",
#                     "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray007.dcm",
#                     "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray008.dcm",
#                     "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray010.dcm",
#                     "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray014.dcm"]


def main(*, cache_directory: str, load_cached: bool = False, save_to_cache: bool = True, plot_first: bool = False,
         max_sinogram_size: int | None = None):
    count: int = len(CT_PATHS)
    # assert len(XRAY_DICOM_PATHS) == count

    i = 0
    run_benchmark(cache_directory=cache_directory, ct_path=CT_PATHS[i], load_cached=load_cached,
                  save_to_cache=save_to_cache, downsample_factor=1, plot=True)


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    # parser.add_argument(
    #     "-p", "--ct-nrrd-path", type=str,
    #     help="Give a path to a NRRD file containing CT data to process. If not provided, some simple "
    #          "synthetic data will be used instead - note that in this case, data will not be saved to "
    #          "the cache.")
    parser.add_argument("-i", "--no-load", action='store_true',
                        help="Do not load any pre-calculated data from the cache.")
    # parser.add_argument(
    #     "-r", "--regenerate-drr", action='store_true',
    #     help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-s", "--max-sinogram-size", type=int, default=None,
                        help="The maximum sinogram size to attempt to calculate. If the '--no-load' flag is not present"
                             ", sinograms of a larger size may still be loaded from the cache. Sinogram size is the "
                             "number of values of r, theta and phi to calculate plane integrals for, and the width and "
                             "height of the grid of samples taken to approximate each integral. The computational "
                             "expense of the 3D Radon transform is O(sinogram_size^5).")
    parser.add_argument("-d", "--display", action='store_true',
                        help="Display/plot the resulting data for the first dataset.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    main(cache_directory=args.cache_directory, load_cached=not args.no_load, save_to_cache=not args.no_save,
             plot_first=args.display, max_sinogram_size=args.max_sinogram_size)
