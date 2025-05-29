import torch
import logging

logger = logging.getLogger(__name__)

from registration.lib.structs import *
from registration.lib.sinogram import *
from registration.lib import grangeat
from registration import data


def calculate_volume_sinogram(cache_directory: str, volume_data: torch.Tensor, *, voxel_spacing: torch.Tensor,
                              ct_volume_path: str, volume_downsample_factor: int, save_to_cache=True,
                              vol_counts=192) -> SinogramClassic:
    device = volume_data.device
    logger.info("Calculating 3D sinogram (the volume to resample)...")

    vol_diag: float = (voxel_spacing * torch.tensor(
        volume_data.size(), dtype=torch.float32, device=voxel_spacing.device)).square().sum().sqrt().item()
    sinogram_range = Sinogram3dRange(
        LinearRange(-.5 * torch.pi, torch.pi * (.5 - 1. / float(vol_counts))),
        LinearRange(-.5 * torch.pi, .5 * torch.pi), LinearRange(-.5 * vol_diag, .5 * vol_diag))

    sinogram3d_grid = Sinogram3dGrid.linear_from_range(sinogram_range, vol_counts, device=device)
    sinogram_data = grangeat.calculate_radon_volume(
        volume_data, voxel_spacing=voxel_spacing, output_grid=sinogram3d_grid, samples_per_direction=vol_counts)

    sinogram3d = SinogramClassic(sinogram_data, sinogram_range)

    logger.info("3D sinogram calculated.")

    if save_to_cache:
        save_path = cache_directory + "/volume_spec_{}.pt".format(data.deterministic_hash(ct_volume_path))
        torch.save(VolumeSpec(ct_volume_path, volume_downsample_factor, sinogram3d), save_path)
        logger.info("3D sinogram saved to '{}'".format(save_path))

    # X, Y, Z = torch.meshgrid(  #     [torch.arange(0, vol_counts, 1), torch.arange(0, vol_counts, 1), torch.arange(
    # 0, vol_counts, 1)])  # fig = pgo.Figure(data=pgo.Volume(x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    # value=sinogram3d.cpu().flatten(),  #                                  isomin=sinogram3d.min().item(),
    # isomax=sinogram3d.max().item(), opacity=.2, surface_count=21))  # fig.show()

    return sinogram3d


def calculate_volume_sinogram_fibonacci(cache_directory: str, volume_data: torch.Tensor, voxel_spacing: torch.Tensor,
                                        ct_volume_path: str, volume_downsample_factor: int, *,
                                        device=torch.device('cpu'), save_to_cache=True,
                                        vol_counts=256) -> SinogramFibonacci:
    logger.info("Calculating 3D Fibonacci sinogram (the volume to resample)...")

    vol_diag: float = (voxel_spacing * torch.tensor(
        volume_data.size(), dtype=torch.float32, device=voxel_spacing.device)).square().sum().sqrt().item()
    r_range = LinearRange(-.5 * vol_diag, .5 * vol_diag)

    sinogram3d_grid = Sinogram3dGrid.fibonacci_from_r_range(r_range, vol_counts, device=device)
    sinogram_data = grangeat.calculate_radon_volume(
        volume_data, voxel_spacing=voxel_spacing, output_grid=sinogram3d_grid, samples_per_direction=vol_counts)

    sinogram3d = SinogramFibonacci(sinogram_data, r_range)

    logger.info("3D Fibonacci sinogram calculated.")

    if save_to_cache:
        save_path = cache_directory + "/volume_spec_fibonacci_{}.pt".format(data.deterministic_hash(ct_volume_path))
        torch.save(VolumeSpecFibonacci(ct_volume_path, volume_downsample_factor, sinogram3d), save_path)
        logger.info("3D Fibonacci sinogram saved to '{}'".format(save_path))

    return sinogram3d
