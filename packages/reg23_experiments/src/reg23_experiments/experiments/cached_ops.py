import logging
import pathlib
import time
from typing import Any, TypedDict, cast

import torch
from beartype import beartype as typechecker
from jaxtyping import Float32, Float64, jaxtyped

from reg23_experiments.data import sinogram
from reg23_experiments.data.structs import Error, LinearRange, Sinogram3dGrid
from reg23_experiments.io import caching
from reg23_experiments.ops import grangeat

__all__ = ["cached_calculate_vif"]

logger = logging.getLogger(__name__)


class _CTModifications(TypedDict):
    downsample_factor: int
    filtering: dict[str, Any]
    truncation_percent: int


class _CachedVifParams(TypedDict):
    sinogram_size: int
    sinogram_type: sinogram.SinogramTypeName
    ct_series_uid: str
    r_range_low: float
    r_range_high: float
    ct_modifications: _CTModifications


class _CachedVif(TypedDict):
    sinogram_data: torch.Tensor


@jaxtyped(typechecker=typechecker)
def cached_calculate_vif(  #
        *,  #
        cache_directory: pathlib.Path,  #
        volume: Float32[torch.Tensor, "p q r"],  #
        voxel_spacing: Float64[torch.Tensor, "3"],  #
        size: int,  #
        sinogram_type: type[sinogram.Sinogram],  #
        ct_series_uid: str,  #
        downsample_factor: int,  #
        filtering: dict[str, Any],  #
        truncation_percent: int,  #
) -> sinogram.Sinogram | Error:
    device = volume.device

    vol_diag: float = (voxel_spacing * torch.tensor(  #
        volume.size(), dtype=torch.float32, device=voxel_spacing.device)).square().sum().sqrt().item()

    r_range = LinearRange(-.5 * vol_diag, .5 * vol_diag)

    params = _CachedVifParams(  #
        sinogram_size=size,  #
        sinogram_type=sinogram.type_to_string(sinogram_type),  #
        ct_series_uid=ct_series_uid,  #
        r_range_low=r_range.low,  #
        r_range_high=r_range.high,  #
        ct_modifications=_CTModifications(  #
            downsample_factor=downsample_factor,  #
            filtering=filtering,  #
            truncation_percent=truncation_percent,  #
        )  #
    )

    # -----
    # Checking cache
    cached: dict[str, Any] | None = caching.load_from_cache(cache_directory=cache_directory, type_name="grangeat_vif",
                                                            params=params)
    if cached is None:
        # -----
        # Calculating a fresh value
        try:
            grid: Sinogram3dGrid = sinogram_type.build_grid(  #
                sinogram_size=params["sinogram_size"],  #
                r_range=r_range,  #
                device=device  #
            )
        except torch.OutOfMemoryError:
            return Error("Insufficient memory to generate {} grid for VIF of size {} on device '{}'.".format(
                params["sinogram_type"], params["sinogram_size"], device))

        logger.info("Calculating Grangeat VIF: volume size = [{} x {} x {}], sinogram size = {}...".format(  #
            volume.size()[0], volume.size()[1], volume.size()[2], params["sinogram_size"]))
        try:
            tic = time.time()
            sinogram_data = grangeat.calculate_radon_volume(  #
                volume,  #
                voxel_spacing=voxel_spacing,  #
                output_grid=grid,  #
                samples_per_direction=params["sinogram_size"]  #
            )
            toc = time.time()
        except MemoryError:
            return Error("Insufficient memory to calculate {} VIF of size {} on device '{}'.".format(  #
                params["sinogram_type"], params["sinogram_size"], volume.device))
        logger.info("Grangeat VIF calculated; took {:.4f}s.".format(toc - tic))

        # -----
        # Saving to cache
        caching.save_to_cache(  #
            cache_directory=cache_directory,  #
            type_name="grangeat_vif",  #
            params=params,  #
            data=_CachedVif(sinogram_data=sinogram_data),  #
        )
    else:
        # -----
        # Taking the cached value
        cached: _CachedVif = cast(_CachedVif, cached)

        sinogram_data = cached["sinogram_data"]

    return sinogram_type(sinogram_data, r_range)
