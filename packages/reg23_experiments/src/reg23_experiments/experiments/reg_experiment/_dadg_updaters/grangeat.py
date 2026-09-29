import logging
import math
import pathlib
from typing import Any

import torch
from jaxtyping import Float64

from reg23_experiments.data import sinogram
from reg23_experiments.data.structs import Error, LinearRange, SceneGeometry, Sinogram2dGrid, Sinogram2dRange, \
    Transformation
from reg23_experiments.experiments.cached_ops import cached_calculate_vif
from reg23_experiments.experiments.helpers import serialise_filter_method, string_to_sim_met
from reg23_experiments.ops import grangeat
from reg23_experiments.ops.data_manager import dadg_updater

__all__ = ["refresh_vif", "refresh_sinogram2d_grid", "refresh_sinogram2d", "resample_for_moving_image_grangeat",
           "apply_sim_metric_grangeat"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["sinogram_size", "vif"])
def refresh_vif(  #
        *,  #
        fixed_sinogram_size: int,  #
        cache_directory: str,  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: torch.Tensor,  #
        sinogram_type: type[sinogram.Sinogram],  #
        ct_series_uid: str,  #
        downsample_level: int,  #
        filter_method: str,  #
        lowpass_threshold: float,  #
        highpass_threshold: float,  #
        truncation_percent: int,  #
) -> dict[str, Any] | Error:
    this_sinogram_size = int(
        math.ceil(pow(ct_volumes[0].numel(), 1.0 / 3.0))) if fixed_sinogram_size is None else fixed_sinogram_size

    downsample_factor = int(2 ** downsample_level)
    downsampled_sinogram_size = this_sinogram_size // downsample_factor

    vif: sinogram.Sinogram | Error = cached_calculate_vif(  #
        cache_directory=pathlib.Path(cache_directory),  #
        volume=ct_volumes[downsample_level],  #
        voxel_spacing=ct_spacing * float(downsample_factor),  #
        size=downsampled_sinogram_size,  #
        sinogram_type=sinogram_type,  #
        ct_series_uid=ct_series_uid,  #
        downsample_factor=downsample_factor,  #
        filtering=serialise_filter_method(  #
            filter_method=filter_method,  #
            lowpass_threshold=lowpass_threshold,  #
            highpass_threshold=highpass_threshold  #
        ),  #
        truncation_percent=truncation_percent,  #
    )

    if isinstance(vif, Error):
        return Error(
            f"Failed to create VIF at level {downsample_level} of type {sinogram_type.__name__}: {vif.description}")

    return {"sinogram_size": this_sinogram_size, "vif": vif}


@dadg_updater(names_returned=["sinogram2d_grid_unshifted", "sinogram2d_grid"])
def refresh_sinogram2d_grid(  #
        *,  #
        cropped_target: torch.Tensor,  #
        fixed_image_offset: torch.Tensor,  #
        fixed_image_spacing: torch.Tensor,  #
) -> dict[str, Any]:
    device = cropped_target.device
    assert fixed_image_offset.device == device
    assert fixed_image_spacing.device == device

    cropped_target_size = cropped_target.size()
    sinogram2d_counts = max(cropped_target_size[0], cropped_target_size[1])
    image_diag: float = (fixed_image_spacing.flip(dims=(0,)) *  #
                         torch.tensor(cropped_target_size, device=device)).square().sum().sqrt().item()
    sinogram2d_range = Sinogram2dRange(LinearRange(-.5 * torch.pi, .5 * torch.pi),
                                       LinearRange(-.5 * image_diag, .5 * image_diag))
    sinogram2d_grid_unshifted = Sinogram2dGrid.linear_from_range(sinogram2d_range, sinogram2d_counts, device=device)

    sinogram2d_grid = sinogram2d_grid_unshifted.shifted(-fixed_image_offset)

    return {"sinogram2d_grid_unshifted": sinogram2d_grid_unshifted, "sinogram2d_grid": sinogram2d_grid}


@dadg_updater(names_returned=["sinogram2d"])
def refresh_sinogram2d(  #
        *,  #
        fixed_image: torch.Tensor,  #
        source_distance: float,  #
        fixed_image_spacing: torch.Tensor,  #
        sinogram2d_grid_unshifted: Sinogram2dGrid,  #
) -> dict[str, Any]:
    sinogram2d = grangeat.calculate_fixed_image(  #
        fixed_image,  #
        source_distance=source_distance,  #
        detector_spacing=fixed_image_spacing,  #
        output_grid=sinogram2d_grid_unshifted,  #
    )

    return {"sinogram2d": sinogram2d}


@dadg_updater(names_returned=["moving_image_grangeat"])
def resample_for_moving_image_grangeat(  #
        *,  #
        current_transformation: Transformation,  #
        vif: sinogram.Sinogram,  #
        sinogram2d_grid: sinogram.Sinogram2dGrid,  #
        source_distance: float,  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
) -> dict[str, Any]:
    device = vif.device
    scene_geometry = SceneGeometry(source_distance=source_distance, fixed_image_offset=fixed_image_offset)
    p_matrix = SceneGeometry.projection_matrix(source_position=scene_geometry.source_position(device=device))

    ph_matrix: torch.Tensor = torch.matmul(  #
        p_matrix,  #
        current_transformation.with_translation_offset(translation_offset).get_h(device=device)  #
    ).to(dtype=torch.float32)

    resampled: torch.Tensor = vif.resample(ph_matrix, sinogram2d_grid)
    return {"moving_image_grangeat": resampled}


@dadg_updater(names_returned=["of_value_grangeat"])
def apply_sim_metric_grangeat(  #
        *,  #
        sim_metric: str,  #
        moving_image_grangeat: torch.Tensor,  #
        sinogram2d: torch.Tensor,  #
) -> dict[str, Any]:
    return {  #
        "of_value_grangeat": -string_to_sim_met(sim_metric)(moving_image_grangeat, sinogram2d),  #
    }
