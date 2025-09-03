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

import logs_setup
from registration.lib import geometry
from registration import data
from registration.lib import sinogram
from registration import pre_computed
from registration.lib.structs import *
from registration.lib import grangeat
from registration import plot_data

import Extension as reg23


def run_benchmark(cache_directory: str, ct_path: str | pathlib.Path, xray_dicom_path: str | pathlib.Path,
                  load_cached: bool = False, save_to_cache: bool = True, downsample_factor: int = 1,
                  sinogram_type: Type[sinogram.SinogramType] = sinogram.SinogramClassic, plot: bool = False,
                  max_sinogram3d_size: int | None = None) -> plot_data.DrrVsGrangeatPlotData.Dataset | None:
    device = torch.device("cuda")

    # Load volumes with a number of downsample factors, and sinograms with a number of sizes
    volume, volume_spacing = data.load_volume(pathlib.Path(ct_path), downsample_factor=downsample_factor)
    volume = volume.to(device=device, dtype=torch.float32)

    sinogram3d_size: int = int(math.ceil(pow(volume.numel(), 1.0 / 3.0)))
    sinogram3d: None | sinogram.Sinogram = None

    loaded_volume_info = None
    if load_cached:
        sinogram_hash = data.deterministic_hash_sinogram(ct_path, sinogram_type, sinogram3d_size, downsample_factor)
        loaded_volume_info = data.load_cached_volume(cache_directory, sinogram_hash)
        if loaded_volume_info is None and max_sinogram3d_size is not None and sinogram3d_size > max_sinogram3d_size:
            sinogram3d_size = max_sinogram3d_size
            sinogram_hash = data.deterministic_hash_sinogram(ct_path, sinogram_type, sinogram3d_size, downsample_factor)
            loaded_volume_info = data.load_cached_volume(cache_directory, sinogram_hash)

    if loaded_volume_info is not None:
        _, sinogram3d = loaded_volume_info

    if sinogram3d is None:
        gc.collect()
        torch.cuda.empty_cache()
        res = pre_computed.calculate_volume_sinogram(cache_directory, volume, voxel_spacing=volume_spacing,
                                                     ct_volume_path=ct_path, volume_downsample_factor=downsample_factor,
                                                     save_to_cache=save_to_cache, sinogram_size=sinogram3d_size,
                                                     sinogram_type=sinogram_type)
        if res is None:
            return None
        sinogram3d, sinogram_evaluation_time = res
        with open("data/sinogram3d_evaluation_times.txt", "a") as file:
            file.write("\n{} {} {}".format(sinogram_type.__name__, sinogram3d_size, sinogram_evaluation_time))

    # Load the X-ray images
    x_ray, detector_spacing, scene_geometry = data.read_dicom(xray_dicom_path)
    x_ray = x_ray.to(device=device)
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool2d(downsample_factor)
        x_ray = down_sampler(x_ray[None, :, :])[0]
        detector_spacing *= downsample_factor

    sinogram2d_size: int = int(math.ceil(pow(x_ray.numel(), 0.5)))
    image_diag: float = (
            detector_spacing * torch.tensor(x_ray.size(), dtype=torch.float32)).square().sum().sqrt().item()
    sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * image_diag, .5 * image_diag))
    sinogram2d_grid = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_size, device=device)
    logger.info(
        "Calculating 2D sinogram (the fixed image): X-ray size = [{} x {}], counts = {}...".format(x_ray.size()[0],
                                                                                                   x_ray.size()[1],
                                                                                                   sinogram2d_size))
    torch.cuda.synchronize()
    tic = time.time()
    x_ray_sinogram = grangeat.calculate_fixed_image(x_ray, source_distance=scene_geometry.source_distance,
                                                    detector_spacing=detector_spacing, output_grid=sinogram2d_grid)
    torch.cuda.synchronize()
    toc = time.time()
    x_ray_sinogram_evaluation_time = toc - tic
    logger.info("X-ray sinogram calculated; took {:.4f}s".format(x_ray_sinogram_evaluation_time))
    with open("data/sinogram2d_evaluation_times.txt", "a") as file:
        file.write("\n{} {}".format(sinogram2d_size, x_ray_sinogram_evaluation_time))
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(x_ray_sinogram.cpu().numpy())
        axes.axis('square')
        axes.set_title("R2 [g^tilde]")
        axes.set_xlabel("r_pol")
        axes.set_ylabel("phi_pol")
        plt.colorbar(mesh)

    # Measure evaluation time of DRR vs grangeat resampling
    repeats: int = 30
    assert repeats >= 1
    transformations = [Transformation.random(device=device) for _ in range(repeats)]
    p_matrix = SceneGeometry.projection_matrix(source_position=scene_geometry.source_position())
    ph_matrices = [torch.matmul(p_matrix, transformation.get_h()).to(dtype=torch.float32, device=device) for
                   transformation in transformations]
    h_matrix_invs = [transformation.inverse().get_h(device=device) for transformation in transformations]
    output_width = x_ray.size()[1]
    output_height = x_ray.size()[0]

    # -----
    # DRR
    # -----
    logger.info("Projecting DRR...")
    torch.cuda.synchronize()
    tic = time.time()
    for i in range(repeats):
        drr = reg23.project_drr(volume, volume_spacing, h_matrix_invs[i], scene_geometry.source_distance, output_width,
                                output_height, torch.zeros(2, dtype=torch.float64), detector_spacing)
    torch.cuda.synchronize()
    toc = time.time()
    drr_evaluation_time: float = (toc - tic) / float(repeats)
    logger.info("DRR projected; took {:.4f}s".format(drr_evaluation_time))
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(drr.cpu().numpy())
        axes.set_title("DRR")
        plt.colorbar(mesh)

    logger.info("Projecting DRR in python...")
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

    # -----
    # DRR mask
    # -----
    logger.info("Projecting DRR mask...")
    torch.cuda.synchronize()
    tic = time.time()
    for i in range(repeats):
        mask = reg23.project_drr_cuboid_mask(torch.tensor(volume.size(), device=device).flip(dims=(0,)),
                                             voxel_spacing=volume_spacing.to(device=device),
                                             homography_matrix_inverse=h_matrix_invs[i].to(device=device),
                                             source_distance=scene_geometry.source_distance, output_width=output_width,
                                             output_height=output_height,
                                             output_offset=torch.zeros(2, dtype=torch.float64, device=device),
                                             detector_spacing=detector_spacing.to(device=device))
    torch.cuda.synchronize()
    toc = time.time()
    mask_evaluation_time: float = (toc - tic) / float(repeats)
    logger.info("DRR mask projected; took {:.4f}s".format(mask_evaluation_time))
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(mask.cpu().numpy())
        axes.set_title("DRR Mask")
        plt.colorbar(mesh)

    logger.info("Projecting DRR mask in python...")
    torch.cuda.synchronize()
    tic = time.time()
    for i in range(repeats):
        mask = geometry.generate_drr_python(volume, voxel_spacing=volume_spacing.to(device=volume.device),
                                            transformation=transformations[i].to(device=volume.device),
                                            scene_geometry=scene_geometry,
                                            output_size=torch.Size([output_height, output_width]),
                                            detector_spacing=detector_spacing, get_ray_intersection_fractions=True)[1]
    torch.cuda.synchronize()
    toc = time.time()
    mask_evaluation_time: float = (toc - tic) / float(repeats)
    logger.info("DRR mask projected in python; took {:.4f}s".format(mask_evaluation_time))
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(mask.cpu().numpy())
        axes.set_title("DRR Mask Python")
        plt.colorbar(mesh)

    # -----
    # Grangeat
    # -----
    logger.info("Resampling sinogram...")
    torch.cuda.synchronize()
    tic = time.time()
    for i in range(repeats):
        resampling = sinogram3d.resample_cuda_texture(ph_matrices[i], sinogram2d_grid)
    torch.cuda.synchronize()
    toc = time.time()
    resampling_time: float = (toc - tic) / float(repeats)
    logger.info("Sinogram resampled; took {:.4f}s".format(resampling_time))
    if plot:
        _, axes = plt.subplots()
        mesh = axes.pcolormesh(resampling.cpu().numpy())
        axes.axis('square')
        axes.set_title("Resampling")
        plt.colorbar(mesh)
        plt.show()

    return plot_data.DrrVsGrangeatPlotData.Dataset(ct_volume_numel=volume.numel(), sinogram3d_size=sinogram3d_size,
                                                   sinogram_type=sinogram_type, x_ray_numel=x_ray.numel(),
                                                   sinogram2d_size=sinogram2d_size, drr_time=drr_evaluation_time,
                                                   resample_time=resampling_time,
                                                   grangeat_fixed_image_time=x_ray_sinogram_evaluation_time)


CT_PATHS = ["/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct001",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct002",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct003",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct005",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct007",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct008",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct010",
            "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/cts_collected/ct014"]
XRAY_DICOM_PATHS = ["/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray001.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray002.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray003.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray005.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray007.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray008.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray010.dcm",
                    "/home/eprager/.local/share/Cryptomator/mnt/Cochlea/xrays_collected/xray014.dcm"]


def main(*, cache_directory: str, load_cached: bool = False, save_to_cache: bool = True, plot_first: bool = False,
         max_sinogram_size: int | None = None):
    count: int = len(CT_PATHS)
    assert len(XRAY_DICOM_PATHS) == count

    if not torch.cuda.is_available():
        logger.error("CUDA not available; cannot run any useful benchmarks.")
        exit(1)

    downsample_factors = [2]  # [1, 2, 4, 8]
    sinogram_types = [sinogram.SinogramClassic]  # , sinogram.SinogramHEALPix]

    datasets: list[plot_data.DrrVsGrangeatPlotData.Dataset] = []
    for sinogram_type in sinogram_types:
        first: bool = True
        for i in range(1):  # count):
            for downsample_factor in downsample_factors:
                res = run_benchmark(cache_directory, CT_PATHS[i], XRAY_DICOM_PATHS[i], load_cached, save_to_cache,
                                    downsample_factor, sinogram_type=sinogram_type, plot=plot_first and first,
                                    max_sinogram3d_size=max_sinogram_size)
                if res is not None:
                    datasets.append(res)
                    first = False
    pdata = plot_data.DrrVsGrangeatPlotData(datasets=datasets)

    torch.save(pdata, "data/drr_vs_grangeat.pkl")


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
