from typing import Tuple

import torch

from registration import data
from registration import pre_computed


def get_volume_and_sinogram(ct_volume_path: str | None, cache_directory: str, *, load_cached: bool, save_to_cache: bool,
                            sinogram_size: int, device) -> Tuple:
    volume_spec = None
    sinogram3d = None
    if load_cached and ct_volume_path is not None:
        volume_spec = data.load_cached_volume(cache_directory, ct_volume_path)

    if volume_spec is None:
        volume_downsample_factor: int = 2
    else:
        volume_downsample_factor, sinogram3d = volume_spec

    if ct_volume_path is None:
        save_to_cache = False
        vol_data = torch.zeros((3, 3, 3), device=device)
        vol_data[1, 1, 1] = 1.
        voxel_spacing = torch.tensor([10., 10., 10.])
    else:
        vol_data, voxel_spacing = data.read_nrrd(ct_volume_path, downsample_factor=volume_downsample_factor)
        vol_data = vol_data.to(device=device, dtype=torch.float32)

    if sinogram3d is None:
        sinogram3d = pre_computed.calculate_volume_sinogram(cache_directory, vol_data, voxel_spacing=voxel_spacing,
                                                            ct_volume_path=ct_volume_path,
                                                            volume_downsample_factor=volume_downsample_factor,
                                                            save_to_cache=save_to_cache, vol_counts=sinogram_size)

    voxel_spacing = voxel_spacing.to(device=device)

    return vol_data, voxel_spacing, sinogram3d
