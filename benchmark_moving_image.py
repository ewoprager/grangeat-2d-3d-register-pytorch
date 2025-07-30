import argparse
import gc
import os
import time
import pickle
import math

import pathlib
import torch
import numpy as np

import logs_setup
from registration import script
from registration import data
from registration.lib import sinogram
from registration import pre_computed
from registration.lib.structs import *
from registration.lib import grangeat
from registration import plot_data

import Extension as reg23


def run_benchmark(cache_directory: str, ct_path: str | pathlib.Path, xray_dicom_path: str | pathlib.Path,
                  load_cached: bool = False, save_to_cache: bool = True,
                  downsample_factor: int = 1) -> plot_data.DrrVsGrangeatPlotData.Dataset:
    device = torch.device("cuda")

    # Load volumes with a number of downsample factors, and sinograms with a number of sizes
    volume, volume_spacing = data.load_volume(pathlib.Path(ct_path), downsample_factor=downsample_factor)
    volume = volume.to(device=device, dtype=torch.float32)

    sinogram3d_size: int = int(math.ceil(pow(volume.numel(), 1.0 / 3.0)))
    sinogram3d: None | torch.Tensor = None
    sinogram_evaluation_time: None | float = None

    volume_spec = None
    if load_cached:
        sinogram_hash = data.deterministic_hash_sinogram(ct_path, sinogram.SinogramClassic, sinogram3d_size, 1)
        volume_spec = data.load_cached_volume(cache_directory, sinogram_hash)

    if volume_spec is None:
        volume_downsample_factor: int = 1
    else:
        volume_downsample_factor, sinogram3d = volume_spec

    if sinogram3d is None:
        sinogram3d, sinogram_evaluation_time = pre_computed.calculate_volume_sinogram(cache_directory, volume,
                                                                                      voxel_spacing=volume_spacing,
                                                                                      ct_volume_path=ct_path,
                                                                                      volume_downsample_factor=volume_downsample_factor,
                                                                                      save_to_cache=save_to_cache,
                                                                                      sinogram_size=sinogram3d_size,
                                                                                      sinogram_type=sinogram.SinogramClassic)

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
    tic = time.time()
    x_ray_sinogram = grangeat.calculate_fixed_image(x_ray, source_distance=scene_geometry.source_distance,
                                                    detector_spacing=detector_spacing, output_grid=sinogram2d_grid)
    toc = time.time()
    x_ray_sinogram_evaluation_time = toc - tic
    logger.info("X-ray sinogram calculated; took {:.4f}s".format(x_ray_sinogram_evaluation_time))

    # Measure evaluation time of DRR vs grangeat resampling
    transformation = Transformation.random(device=device)
    p_matrix = SceneGeometry.projection_matrix(source_position=scene_geometry.source_position())
    ph_matrix = torch.matmul(p_matrix, transformation.get_h()).to(dtype=torch.float32, device=device)
    h_matrix_inv = transformation.inverse().get_h().to(device=device)
    output_width = x_ray.size()[1]
    output_height = x_ray.size()[0]
    # DRR:
    logger.info("Projecting DRR...")
    tic = time.time()
    drr = reg23.project_drr(volume, volume_spacing, h_matrix_inv, scene_geometry.source_distance, output_width,
                            output_height, torch.zeros(2, dtype=torch.float64), detector_spacing)
    toc = time.time()
    drr_evaluation_time: float = toc - tic
    logger.info("DRR projected; took {:.4f}s".format(drr_evaluation_time))
    # Grangeat:
    logger.info("Resampling sinogram...")
    tic = time.time()
    resampling = sinogram3d.resample_cuda_texture(ph_matrix, sinogram2d_grid)
    toc = time.time()
    resampling_time: float = toc - tic
    logger.info("Sinogram resampled; took {:.4f}s".format(drr_evaluation_time))

    return plot_data.DrrVsGrangeatPlotData.Dataset(ct_volume_numel=volume.numel(), sinogram3d_size=sinogram3d_size,
                                                   x_ray_numel=x_ray.numel(), sinogram2d_size=sinogram2d_size,
                                                   drr_time=drr_evaluation_time, resample_time=resampling_time)


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


def main(*, cache_directory: str, load_cached: bool = False, save_to_cache: bool = True):
    count: int = len(CT_PATHS)
    assert len(XRAY_DICOM_PATHS) == count

    if not torch.cuda.is_available():
        logger.error("CUDA not available; cannot run any useful benchmarks.")
        exit(1)

    downsample_factors = [1, 2, 4, 8]

    datasets: list[plot_data.DrrVsGrangeatPlotData.Dataset] = []
    for i in range(count):
        for downsample_factor in downsample_factors:
            datasets.append(run_benchmark(cache_directory, CT_PATHS[i], XRAY_DICOM_PATHS[i], load_cached, save_to_cache,
                                          downsample_factor))
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
    # parser.add_argument(
    #     "-s", "--sinogram-size", type=int, default=256,
    #     help="The number of values of r, theta and phi to calculate plane integrals for, "
    #          "and the width and height of the grid of samples taken to approximate each integral. The "
    #          "computational expense of the 3D Radon transform is O(sinogram_size^5).")
    # parser.add_argument(
    #     "-d", "--display", action='store_true', help="Display/plot the resulting data.")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    main(cache_directory=args.cache_directory, load_cached=not args.no_load, save_to_cache=not args.no_save)
