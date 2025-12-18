from typing import Tuple, Type
import pathlib
import math

import torch

from reg23_experiments.registration import data
from reg23_experiments.registration import pre_computed
from reg23_experiments.registration.lib import sinogram

__all__ = ["get_volume_and_sinogram"]


def get_volume_and_sinogram(ct_volume_path: str | None, cache_directory: str, *, load_cached: bool, save_to_cache: bool,
                            sinogram_size: int | None, device,
                            sinogram_type: Type[sinogram.SinogramType] = sinogram.SinogramClassic,
                            volume_downsample_factor: int = 2) -> Tuple | None:
    if ct_volume_path is None:
        save_to_cache = False
        vol_data = torch.zeros((7, 7, 7), device=device)
        # vol_data[0, 0, 0] = 1.
        vol_data[1, 1, 1] = 1.
        vol_data[0, 3, :] = 0.7
        vol_data[6, :, :] = 0.2
        vol_data[3:6, 2, 3] = 0.8
        voxel_spacing = torch.tensor([10., 10., 10.])
    else:
        vol_data, voxel_spacing = data.load_volume(pathlib.Path(ct_volume_path),
                                                   downsample_factor=volume_downsample_factor)
        vol_data = vol_data.to(device=device, dtype=torch.float32)

    if sinogram_size is None:
        sinogram_size = int(math.ceil(pow(vol_data.numel(), 1.0 / 3.0)))

    volume_spec = None
    sinogram3d = None
    if load_cached and ct_volume_path is not None:
        sinogram_hash = data.deterministic_hash_sinogram(ct_volume_path, sinogram_type, sinogram_size,
                                                         volume_downsample_factor)
        volume_spec = data.load_cached_volume(cache_directory, sinogram_hash)

    if volume_spec is not None:
        _, sinogram3d = volume_spec

    if sinogram3d is None:
        res = pre_computed.calculate_volume_sinogram(cache_directory,  #
                                                     vol_data,  #
                                                     voxel_spacing=voxel_spacing,  #
                                                     ct_volume_path=ct_volume_path,  #
                                                     volume_downsample_factor=volume_downsample_factor,  #
                                                     save_to_cache=save_to_cache,  #
                                                     sinogram_size=sinogram_size,  #
                                                     sinogram_type=sinogram_type)
        if res is None:
            return None
        sinogram3d, _ = res

    voxel_spacing = voxel_spacing.to(device=device)

    return vol_data, voxel_spacing, sinogram3d
