from typing import Tuple, NamedTuple
import time
import os
import argparse
import logging.config

import matplotlib.pyplot as plt
import torch

import Extension as ExtensionTest

from registration.lib.structs import *
from registration.lib.sinogram import *
import registration.data as data
import registration.pre_computed as pre_computed
from registration.lib.structs import LinearRange, SceneGeometry


class TaskSummary(NamedTuple):
    name: str
    result: torch.Tensor


class FunctionParams(NamedTuple):
    sinogram3d: SinogramClassic
    ph_matrix: torch.Tensor
    fixed_image_grid: Sinogram2dGrid

    def to(self, **kwargs) -> 'FunctionParams':
        return FunctionParams(
            self.sinogram3d.to(**kwargs), self.ph_matrix.to(**kwargs), self.fixed_image_grid.to(**kwargs))


def task_resample_sinogram3d(function, params: FunctionParams) -> torch.Tensor:
    return function(params.sinogram3d, params.ph_matrix, params.fixed_image_grid)


def plot_task_resample_sinogram3d(summary: TaskSummary):
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(summary.result.clone())
    axes.axis('square')
    axes.set_title("d/dr R3 [mu] resampled: {}".format(summary.name))
    axes.set_xlabel("r")
    axes.set_ylabel("phi")
    plt.colorbar(mesh)


def run_task(task, task_plot, function, name: str, device: str, params: FunctionParams) -> TaskSummary:
    params_device = params.to(device=device)
    logger.info("Running {} on {}...".format(name, device))
    tic = time.time()
    output = task(function, params_device)
    toc = time.time()
    logger.info("Done; took {:.3f}s. Saving and plotting...".format(toc - tic))
    name: str = "{}_on_{}".format(name, device)
    summary = TaskSummary(name, output.cpu())
    torch.save(summary.result, "cache/{}.pt".format(summary.name))
    task_plot(summary)
    logger.info("Done.")
    return summary


def main(*, path: str | None, cache_directory: str, load_cached: bool, sinogram_size: int, save_to_cache: bool = True):
    logger.info("----- Benchmarking resample_sinogram3d -----")

    cuda = torch.device('cuda')

    volume_spec = None
    sinogram3d = None
    if load_cached and path is not None:
        volume_spec = data.load_cached_volume(cache_directory, path)

    if volume_spec is None:
        volume_downsample_factor: int = 1
    else:
        volume_downsample_factor, sinogram3d = volume_spec

    if path is None:
        save_to_cache = False
        vol_data = torch.zeros((3, 3, 3))
        vol_data[1, 1, 1] = 1.
        voxel_spacing = torch.tensor([10., 10., 10.])
    else:
        vol_data, voxel_spacing = data.load_volume(path, downsample_factor=volume_downsample_factor)
        vol_data = vol_data.to(dtype=torch.float32, device=cuda)

    if sinogram3d is None:
        sinogram3d = pre_computed.calculate_volume_sinogram(
            cache_directory, vol_data, voxel_spacing, path, volume_downsample_factor, save_to_cache=save_to_cache,
            vol_counts=sinogram_size)

    vol_diag: float = (
            torch.tensor([vol_data.size()], dtype=torch.float32) * voxel_spacing).square().sum().sqrt().item()

    fixed_image_range = Sinogram2dRange(
        LinearRange(-.5 * torch.pi, .5 * torch.pi), LinearRange(-.5 * vol_diag, .5 * vol_diag))
    fixed_image_grid = Sinogram2dGrid.linear_from_range(fixed_image_range, (1000, 1000))

    scene_geometry = SceneGeometry(source_distance=1000.)
    # transformation = Transformation(rotation=torch.tensor([0., 0., 0.]), translation=torch.tensor([0., 0., 0.]))
    transformation = Transformation.random()
    source_position = scene_geometry.source_position()
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    ph_matrix = torch.matmul(p_matrix, transformation.get_h()).to(dtype=torch.float32)

    params = FunctionParams(sinogram3d, ph_matrix, fixed_image_grid)

    outputs: list[TaskSummary] = [
        run_task(
            task_resample_sinogram3d, plot_task_resample_sinogram3d, SinogramClassic.resample, "Sinogram3D.resample",
            "cpu", params), run_task(
            task_resample_sinogram3d, plot_task_resample_sinogram3d, SinogramClassic.resample, "Sinogram3D.resample",
            "cuda", params), run_task(
            task_resample_sinogram3d, plot_task_resample_sinogram3d, SinogramClassic.resample_python,
            "Sinogram3D.resample_python", "cpu", params), run_task(
            task_resample_sinogram3d, plot_task_resample_sinogram3d, SinogramClassic.resample_python,
            "Sinogram3D.resample_python", "cuda", params)]

    plt.show()

    logger.info("Calculating discrepancies...")
    # found: bool = False
    for i in range(len(outputs) - 1):
        discrepancy = (outputs[i].result - outputs[i + 1].result).abs().mean() / (.5 * (
                outputs[i].result.max() - outputs[i].result.min() + outputs[i + 1].result.max() - outputs[
            i + 1].result.min()))
        # if discrepancy > 1e-2:
        #     found = True
        logger.info(
            "\tAverage discrepancy between outputs {} and {} is {:.3f} %".format(
                outputs[i].name, outputs[i + 1].name, 100. * discrepancy))
    # if not found:
    #     logger.info("\tNo discrepancies found.")
    logger.info("Done.")

    # logger.info("Showing plots...")  # X, Y, Z = torch.meshgrid([torch.arange(0, size[0], 1), torch.arange(0,
    # size[1], 1), torch.arange(0, size[2], 1)])  # fig = pgo.Figure(  #     data=pgo.Volume(x=X.flatten(),
    # y=Y.flatten(), z=Z.flatten(), value=image.flatten(), isomin=.0, isomax=2000.,  #  # opacity=.1,
    # surface_count=21), layout=pgo.Layout(title="Input"))  # fig.show()


if __name__ == "__main__":
    # set up logger
    logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
    logger = logging.getLogger("radonRegistration")

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument(
        "-c", "--cache-directory", type=str, default="cache",
        help="Set the directory where data that is expensive to calculate will be saved. The default "
             "is 'cache'.")
    parser.add_argument(
        "-p", "--ct-nrrd-path", type=str,
        help="Give a path to a NRRD file containing CT data to process. If not provided, some simple "
             "synthetic data will be used instead - note that in this case, data will not be saved to "
             "the cache.")
    parser.add_argument(
        "-i", "--no-load", action='store_true', help="Do not load any pre-calculated data from the cache.")
    parser.add_argument(
        "-r", "--regenerate-drr", action='store_true',
        help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument(
        "-s", "--sinogram-size", type=int, default=256,
        help="The number of values of r, theta and phi to calculate plane integrals for, "
             "and the width and height of the grid of samples taken to approximate each integral. The "
             "computational expense of the 3D Radon transform is O(sinogram_size^5).")
    args = parser.parse_args()

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    main(
        path=args.ct_nrrd_path, cache_directory=args.cache_directory, load_cached=not args.no_load,
        save_to_cache=not args.no_save, sinogram_size=args.sinogram_size)
