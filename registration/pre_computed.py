from typing import Any
import time
import logging

logger = logging.getLogger(__name__)

import torch

from registration.lib.structs import *
from registration.lib.sinogram import *
from registration.lib import grangeat
from registration import data


def calculate_volume_sinogram(cache_directory: str, volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor,
                              ct_volume_path: str, volume_downsample_factor: int, save_to_cache=True, sinogram_size=192,
                              sinogram_type: Type[SinogramType] = SinogramClassic) -> Tuple[SinogramType, float]:
    device = volume_data.device

    vol_diag: float = (voxel_spacing * torch.tensor(
        volume_data.size(), dtype=torch.float32, device=voxel_spacing.device)).square().sum().sqrt().item()
    r_range = LinearRange(-.5 * vol_diag, .5 * vol_diag)

    if sinogram_type == SinogramClassic:
        sinogram3d_grid = SinogramClassic.build_grid(counts=sinogram_size, r_range=r_range, device=device)
    elif sinogram_type == SinogramHEALPix:
        sinogram3d_grid = SinogramHEALPix.build_grid(
            n_side=int(torch.ceil(torch.tensor(float(sinogram_size)) / torch.tensor(6.).sqrt()).item()),
            r_count=sinogram_size, r_range=r_range, device=device)
    else:
        raise TypeError(
            "Unrecognised sinogram type '{}.{}'".format(sinogram_type.__module__, sinogram_type.__qualname__))

    logger.info(
        "Calculating 3D sinogram (the volume to resample): volume size = [{} x {} x {}], sinogram size = {}...".format(
            volume_data.size()[0], volume_data.size()[1], volume_data.size()[2], sinogram_size))
    tic = time.time()
    sinogram_data = grangeat.calculate_radon_volume(
        volume_data, voxel_spacing=voxel_spacing, output_grid=sinogram3d_grid, samples_per_direction=sinogram_size)
    toc = time.time()
    sinogram_evaluation_time: float = toc - tic
    logger.info("3D sinogram calculated; took {:.4f}s.".format(sinogram_evaluation_time))

    sinogram3d = sinogram_type(sinogram_data, r_range)

    if save_to_cache:
        save_path = cache_directory + "/volume_spec_{}.pt".format(
            data.deterministic_hash_sinogram(ct_volume_path, sinogram_type, sinogram_size, volume_downsample_factor))
        torch.save(VolumeSpec(ct_volume_path, volume_downsample_factor, sinogram3d), save_path)
        logger.info("3D sinogram saved to '{}'".format(save_path))

    # X, Y, Z = torch.meshgrid(  #     [torch.arange(0, sinogram_size, 1), torch.arange(0, sinogram_size, 1),
    # torch.arange(
    # 0, sinogram_size, 1)])  # fig = pgo.Figure(data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    # value=sinogram3d.cpu().flatten(),  #                                  isomin=sinogram3d.min().item(),
    # isomax=sinogram3d.max().item(), opacity=.2, surface_count=21))  # fig.show()

    return sinogram3d, sinogram_evaluation_time

# def calculate_volume_sinogram_fibonacci(cache_directory: str, volume_data: torch.Tensor, voxel_spacing: torch.Tensor,
#                                         ct_volume_path: str, volume_downsample_factor: int, *,
#                                         device=torch.device('cpu'), save_to_cache=True,
#                                         sinogram_size=256) -> SinogramFibonacci:
#     logger.info("Calculating 3D Fibonacci sinogram (the volume to resample)...")
#
#     vol_diag: float = (voxel_spacing * torch.tensor(
#         volume_data.size(), dtype=torch.float32, device=voxel_spacing.device)).square().sum().sqrt().item()
#     r_range = LinearRange(-.5 * vol_diag, .5 * vol_diag)
#
#     sinogram3d_grid = Sinogram3dGrid.fibonacci_from_r_range(r_range, sinogram_size, device=device)
#     sinogram_data = grangeat.calculate_radon_volume(
#         volume_data, voxel_spacing=voxel_spacing, output_grid=sinogram3d_grid, samples_per_direction=sinogram_size)
#
#     sinogram3d = SinogramFibonacci(sinogram_data, r_range)
#
#     logger.info("3D Fibonacci sinogram calculated.")
#
#     if save_to_cache:
#         save_path = cache_directory + "/volume_spec_fibonacci_{}.pt".format(data.deterministic_hash(ct_volume_path))
#         torch.save(VolumeSpecFibonacci(ct_volume_path, volume_downsample_factor, sinogram3d), save_path)
#         logger.info("3D Fibonacci sinogram saved to '{}'".format(save_path))
#
#     return sinogram3d
