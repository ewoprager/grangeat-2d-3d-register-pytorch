import argparse
import os
import logging.config

import matplotlib.pyplot as plt
import numpy as np
import torch
import nrrd

import Extension

from registration.lib.structs import *
from registration.lib.sinogram import *
from registration import drr
from registration.lib import geometry
from registration import data
from registration import pre_computed
from registration import objective_function
from registration import script
import registration.lib.plot as myplt


def main(*, path: str | None, cache_directory: str, load_cached: bool, regenerate_drr: bool, save_to_cache: bool,
         sinogram_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    # Load the volume and get its sinogram
    vol_data, voxel_spacing, sinogram3d = script.get_volume_and_sinogram(path, cache_directory, load_cached=load_cached,
                                                                         save_to_cache=save_to_cache,
                                                                         sinogram_size=sinogram_size, device=device)

    # Load / generate a DRR through the volume
    drr_spec = None
    if not regenerate_drr:
        drr_spec = data.load_cached_drr(cache_directory, path)

    if drr_spec is None:
        drr_spec = drr.generate_new_drr(cache_directory, path, vol_data, voxel_spacing, device=device,
                                        save_to_cache=save_to_cache)

    detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation_ground_truth = drr_spec

    logger.info("Plotting DRR and fixed image...")
    # Plotting DRR
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(drr_image.cpu())
    axes.axis('square')
    axes.set_title("g")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    plt.colorbar(mesh)

    # nrrd.write("/home/eprager/Desktop/projection_image.nrrd", drr_image.cpu().unsqueeze(0).numpy())

    # Plotting fixed image (d/ds R_2 [g^tilde])
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(fixed_image.cpu())
    axes.axis('square')
    axes.set_title("d/ds R2 [g^tilde]")
    axes.set_xlabel("r")
    axes.set_ylabel("phi")
    plt.colorbar(mesh)
    # Getting the limits to use for other colour mesh plots:
    colour_limits: Tuple[float, float] = mesh.get_clim()
    logger.info("DRR and fixed image plotted.")

    sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, fixed_image.size(), device=device)

    logger.info("Evaluating at ground truth...")
    zncc, resampled = objective_function.evaluate(fixed_image, sinogram3d,
                                                  transformation=transformation_ground_truth.to(device=device),
                                                  scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid,
                                                  plot=colour_limits)
    logger.info("Evaluation: -ZNCC = -{:.4e}".format(zncc.item()  # evaluate_direct(fixed_image, vol_data,
                                                     # transformation=transformation_ground_truth,
                                                     #                 scene_geometry=scene_geometry,
                                                     #                 fixed_image_grid=sinogram2d_grid,
                                                     #                 voxel_spacing=voxel_spacing,
                                                     #                 plot=True)
                                                     ))

    logger.info("Evaluating at ground truth with sample smoothing...")
    zncc, resampled = objective_function.evaluate(fixed_image, sinogram3d,
                                                  transformation=transformation_ground_truth.to(device=device),
                                                  scene_geometry=scene_geometry, fixed_image_grid=sinogram2d_grid,
                                                  plot=colour_limits, smooth=True)
    logger.info("Evaluation with sample smoothing, -ZNCC = -{:.4e}".format(zncc.item()))

    plt.show()


if __name__ == "__main__":
    # set up logger
    logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
    logger = logging.getLogger("radonRegistration")

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    parser.add_argument("-p", "--ct-nrrd-path", type=str,
                        help="Give a path to a NRRD file containing CT data to process. If not provided, some simple "
                             "synthetic data will be used instead - note that in this case, data will not be saved to "
                             "the cache.")
    parser.add_argument("-i", "--no-load", action='store_true',
                        help="Do not load any pre-calculated data from the cache.")
    parser.add_argument("-r", "--regenerate-drr", action='store_true',
                        help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-s", "--sinogram-size", type=int, default=256,
                        help="The number of values of r, theta and phi to calculate plane integrals for, "
                             "and the width and height of the grid of samples taken to approximate each integral. The "
                             "computational expense of the 3D Radon transform is O(sinogram_size^5).")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    main(path=args.ct_nrrd_path, cache_directory=args.cache_directory, load_cached=not args.no_load,
         regenerate_drr=args.regenerate_drr, save_to_cache=not args.no_save, sinogram_size=args.sinogram_size)
