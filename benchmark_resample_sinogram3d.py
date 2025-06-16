from typing import Tuple, NamedTuple, Type, TypeVar
import time
import os
import argparse
import logging.config

import matplotlib.pyplot as plt
import torch
import pathlib
import plotly.graph_objects as pgo

from registration.lib.structs import *
from registration.lib import sinogram
import registration.data as data
import registration.pre_computed as pre_computed
from registration import script


class TaskSummary(NamedTuple):
    name: str
    result: torch.Tensor


class FunctionParams(NamedTuple):
    sinogram3d: sinogram.Sinogram
    fixed_image_grid: Sinogram2dGrid

    def to(self, **kwargs) -> 'FunctionParams':
        return FunctionParams(
            self.sinogram3d.to(**kwargs), self.fixed_image_grid.to(**kwargs))


def task_resample_sinogram3d(function, params: FunctionParams, ph_matrix: torch.Tensor) -> torch.Tensor:
    return function(params.sinogram3d, ph_matrix, params.fixed_image_grid)


def plot_task_resample_sinogram3d(summary: TaskSummary):
    _, axes = plt.subplots()
    mesh = axes.pcolormesh(summary.result.clone())
    axes.axis('square')
    axes.set_title("d/dr R3 [mu] resampled: {}".format(summary.name))
    axes.set_xlabel("r")
    axes.set_ylabel("phi")
    plt.colorbar(mesh)


def random_ph_matrix() -> torch.Tensor:
    scene_geometry = SceneGeometry(source_distance=1000.)
    transformation = Transformation.random()
    source_position = scene_geometry.source_position()
    p_matrix = SceneGeometry.projection_matrix(source_position=source_position)
    return torch.matmul(p_matrix, transformation.get_h()).to(dtype=torch.float32)


def run_task(task, task_plot, function, name: str, device: str, params: FunctionParams,
             ph_matrices: list[torch.Tensor]) -> TaskSummary:
    params_device = params.to(device=device)
    logger.info("Running {} on {}...".format(name, device))
    summed = 0.0
    for ph_matrix in ph_matrices[:-1]:
        tic = time.time()
        task(function, params_device, ph_matrix)
        toc = time.time()
        summed += toc - tic
    tic = time.time()
    output = task(function, params_device, ph_matrices[-1])
    toc = time.time()
    summed += toc - tic
    logger.info(
        "Done; took {:.5f}s total; {:.5f}s per evaluation. Saving and plotting...".format(
            summed, summed / float(len(ph_matrices))))
    name: str = "{}_on_{}".format(name, device)
    summary = TaskSummary(name, output.cpu())
    torch.save(summary.result, "cache/{}.pt".format(summary.name))
    task_plot(summary)
    logger.info("Done.")
    return summary


def main(*, path: str | None, cache_directory: str, load_cached: bool, sinogram_size: int, save_to_cache: bool = True):
    logger.info("----- Benchmarking resample_sinogram3d -----")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ph_matrices_cpu = [random_ph_matrix() for _ in range(1)]
    if device.type == "cuda":
        ph_matrices_cuda = [m.to(device=device) for m in ph_matrices_cpu]

    sinogram_types = [sinogram.SinogramClassic, sinogram.SinogramHEALPix]
    outputs: list[TaskSummary] = []
    for sinogram_type in sinogram_types:

        vol_data, voxel_spacing, sinogram3d = script.get_volume_and_sinogram(
            path, cache_directory, load_cached=load_cached, save_to_cache=save_to_cache, sinogram_size=sinogram_size,
            sinogram_type=sinogram_type, device=device, volume_downsample_factor=32)

        plot_radon_volume: bool = False
        if plot_radon_volume:
            size = sinogram3d.data.size()
            X, Y, Z = torch.meshgrid(
                [torch.arange(0, size[0], 1), torch.arange(0, size[1], 1), torch.arange(0, size[2], 1)])
            fig = pgo.Figure(
                data=pgo.Volume(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=sinogram3d.data.cpu().flatten(),
                    isomin=sinogram3d.data.min().item(), isomax=sinogram3d.data.max().item(), opacity=.2,
                    surface_count=21), layout=pgo.Layout(title="Sinogram"))
            fig.show()

        vol_diag: float = (torch.tensor(
            [vol_data.size()], dtype=torch.float32,
            device=voxel_spacing.device) * voxel_spacing).square().sum().sqrt().item()

        fixed_image_range = Sinogram2dRange(
            LinearRange(-.5 * torch.pi, .5 * torch.pi), LinearRange(-.5 * vol_diag, .5 * vol_diag))
        fixed_image_grid = Sinogram2dGrid.linear_from_range(fixed_image_range, (1000, 1000))

        params = FunctionParams(sinogram3d, fixed_image_grid)

        # outputs.append(
        #     run_task(
        #         task_resample_sinogram3d, plot_task_resample_sinogram3d, sinogram_type.resample_python,
        #         "{}.resample_python".format(sinogram_type.__name__), "cpu", params, ph_matrices_cpu))
        if device.type == "cuda":
            outputs.append(
                run_task(
                    task_resample_sinogram3d, plot_task_resample_sinogram3d, sinogram_type.resample_python,
                    "{}.resample_python".format(sinogram_type.__name__), "cuda", params, ph_matrices_cuda))
        run_extension: bool = True
        if run_extension:
            # outputs.append(
            #     run_task(
            #         task_resample_sinogram3d, plot_task_resample_sinogram3d, sinogram_type.resample,
            #         "{}.resample".format(sinogram_type.__name__), "cpu", params, ph_matrices_cpu))
            if device.type == "cuda":
                outputs.append(
                    run_task(
                        task_resample_sinogram3d, plot_task_resample_sinogram3d, sinogram_type.resample,
                        "{}.resample".format(sinogram_type.__name__), "cuda", params, ph_matrices_cuda))

                outputs.append(
                    run_task(
                        task_resample_sinogram3d, plot_task_resample_sinogram3d, sinogram_type.resample_cuda_texture,
                        "{}.resample_cuda_texture".format(sinogram_type.__name__), "cuda", params, ph_matrices_cuda))

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
