import logging
import os
from typing import Any

os.environ["QT_API"] = "PyQt6"

import torch
from jaxtyping import Float32, Float64

import reg23_core
from reg23_experiments.data.segmentation import NamedPoints2D, NamedPoints3D
from reg23_experiments.data.structs import SceneGeometry, Transformation
from reg23_experiments.experiments.helpers import string_to_sim_met
from reg23_experiments.ops import geometry
from reg23_experiments.ops.data_manager import dadg_updater

__all__ = ["apply_sim_metric", "refresh_scaling_image", "refresh_weight_image", "project_drr", "project_fiducials"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["scaling_image", "fixed_image"])
def refresh_scaling_image(  #
        *,  #
        current_transformation: Transformation,  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        source_distance: float,  #
        fixed_image_spacing: Float64[torch.Tensor, "2"],  #
        fixed_image_size: torch.Size,  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
        cropped_target: Float32[torch.Tensor, "n m"],  #
) -> dict[str, Any]:
    h_inv: torch.Tensor = current_transformation.with_translation_offset(translation_offset).inverse().get_h(
        device=ct_volumes[0].device)
    scaling_image = reg23_core.project_drr_cuboid_mask(  #
        volume_size=torch.tensor(ct_volumes[0].size(), device=ct_volumes[0].device).flip(dims=(0,)),  #
        voxel_spacing=ct_spacing,  #
        homography_matrix_inverse=h_inv,  #
        source_distance=source_distance,  #
        output_width=fixed_image_size[1],  #
        output_height=fixed_image_size[0],  #
        output_offset=fixed_image_offset,  #
        detector_spacing=fixed_image_spacing  #
    )
    # Generate the fixed images
    fixed_image = cropped_target
    return {  #
        "scaling_image": scaling_image,  #
        "fixed_image": fixed_image,  #
    }


def refresh_weight_image(  #
        *,  #
        apply_weighting: bool,  #
        weight_alpha: float,  #
        scaling_image: Float32[torch.Tensor, "n m"],  #
        weight_epsilon: float = 1e-5,  #
) -> dict[str, Any]:
    if apply_weighting:
        if weight_alpha < 1e-2:
            weight_image = scaling_image.clone()
            weight_image[weight_image < 1.0 - weight_epsilon] = 0.0
        else:
            sc_sq = scaling_image.square()
            weight_image = torch.pow(3.0 * sc_sq - 2.0 * scaling_image * sc_sq, 1.0 / (weight_alpha * weight_alpha))
    else:
        weight_image = None
    return {  #
        "weight_image": weight_image,  #
    }


@dadg_updater(names_returned=["moving_image"])
def project_drr(  #
        *,  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        current_transformation: Transformation,  #
        fixed_image_size: torch.Size,  #
        source_distance: float,  #
        fixed_image_spacing: Float64[torch.Tensor, "2"],  #
        downsample_level: int,  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"]  #
) -> dict[str, Any]:
    return {"moving_image": geometry.generate_drr(  #
        ct_volumes[downsample_level],  #
        transformation=current_transformation.with_translation_offset(translation_offset),  #
        voxel_spacing=ct_spacing * 2.0 ** downsample_level,  #
        detector_spacing=fixed_image_spacing,  #
        scene_geometry=SceneGeometry(source_distance=source_distance, fixed_image_offset=fixed_image_offset),  #
        output_size=fixed_image_size  #
    )}


@dadg_updater(names_returned=["of_value"])
def apply_sim_metric(  #
        *,  #
        sim_metric: str,  #
        moving_image: Float32[torch.Tensor, "n m"],  #
        fixed_image: Float32[torch.Tensor, "n m"],  #
        weight_image: Float32[torch.Tensor, "n m"] | None,  #
) -> dict[str, Any]:
    return {  #
        "of_value": -string_to_sim_met(sim_metric)(moving_image, fixed_image, weights=weight_image),  #
    }


@dadg_updater(names_returned=["projected_fiducials"])
def project_fiducials(  #
        *,  #
        current_transformation: Transformation,  #
        untruncated_ct_volume: Float32[torch.Tensor, "p q r"],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        image_2d_full: Float32[torch.Tensor, "n m"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        image_2d_full_spacing: Float64[torch.Tensor, "2"],  #
        ct_fiducial_points: NamedPoints3D,  #
        source_distance: float,  #
) -> dict[str, Any]:
    device = torch.device("cpu")
    transformation = current_transformation.to(device=device).with_translation_offset(translation_offset)
    input_vectors = ct_fiducial_points.data.cpu() - 0.5 * ct_spacing.cpu() * torch.tensor(untruncated_ct_volume.size(),
                                                                                          dtype=torch.float64).flip(
        dims=(0,))
    projected = geometry.project_vectors(input_vectors, source_distance=source_distance, transformation=transformation)
    size_tensor = torch.tensor(image_2d_full.size(), dtype=torch.float64).flip(dims=(0,))
    output_points_2d = projected + 0.5 * image_2d_full_spacing.cpu() * size_tensor
    return {"projected_fiducials": NamedPoints2D(names=ct_fiducial_points.names, data=output_points_2d)}
