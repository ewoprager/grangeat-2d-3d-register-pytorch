from typing import Any

import torch
from jaxtyping import Float32, Float64

import reg23_core
from reg23_experiments.data.structs import Transformation
from reg23_experiments.experiments.helpers import ParametrisedSimilarityMetric, string_to_sim_met
from reg23_experiments.ops.data_manager import dadg_updater
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation

__all__ = ["refresh_masks", "refresh_weights", "project_moving_images", "apply_sim_metric"]


# @dadg_updater(names_returned=["masks", "fixed_images"])
def refresh_masks(  #
        *,  #
        parameters: Float64[torch.Tensor, "b 6"],  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        source_distance: float,  #
        fixed_image_spacing: Float64[torch.Tensor, "2"],  #
        fixed_image_size: torch.Size,  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
        cropped_target: Float32[torch.Tensor, "n m"],  #
) -> dict[str, Any]:
    ts: list[Transformation] = [mapping_parameters_to_transformation(p) for p in parameters]
    h_invs: torch.Tensor = torch.stack([  #
        t.with_translation_offset(translation_offset).inverse().get_h(device=ct_volumes[0].device)  #
        for t in ts  #
    ], dim=0)
    masks = reg23_core.project_drr_cuboid_masks_batched(  #
        volume_size=torch.tensor(ct_volumes[0].size(), device=ct_volumes[0].device).flip(dims=(0,)),  #
        voxel_spacing=ct_spacing,  #
        inverse_h_matrices=h_invs,  #
        source_distance=source_distance,  #
        output_width=fixed_image_size[1],  #
        output_height=fixed_image_size[0],  #
        output_offset=fixed_image_offset,  #
        detector_spacing=fixed_image_spacing  #
    )
    # Generate the masked fixed images
    return {  #
        "masks": masks,  #
        "fixed_images": masks * cropped_target.unsqueeze(0),  #
    }


def refresh_weights(  #
        *,  #
        weight_alpha: float,  #
        masks: Float32[torch.Tensor, "b n m"],  #
        weight_epsilon: float = 1e-5,  #
) -> dict[str, Any]:
    if weight_alpha < 1e-2:
        weight_images = masks.clone()
        weight_images[weight_images < 1.0 - weight_epsilon] = 0.0
    else:
        masks_sq = masks.square()
        weight_images = torch.pow(3.0 * masks_sq - 2.0 * masks * masks_sq, 1.0 / (weight_alpha * weight_alpha))
    return {  #
        "weight_images": weight_images,  #
    }


@dadg_updater(names_returned=["moving_images"])
def project_moving_images(  #
        *,  #
        parameters: Float64[torch.Tensor, "b 6"],  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        source_distance: float,  #
        fixed_image_size: torch.Size,  #
        fixed_image_spacing: Float64[torch.Tensor, "2"],  #
        downsample_level: int,  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
) -> dict[str, Any]:
    ts: list[Transformation] = [mapping_parameters_to_transformation(p) for p in parameters]
    h_invs: torch.Tensor = torch.stack([  #
        t.with_translation_offset(translation_offset).inverse().get_h(device=ct_volumes[0].device)  #
        for t in ts  #
    ], dim=0)
    return {  #
        "moving_images": reg23_core.project_drrs_batched(  #
            volume=ct_volumes[downsample_level],  #
            voxel_spacing=ct_spacing * 2.0 ** downsample_level,  #
            inverse_h_matrices=h_invs,  #
            source_distance=source_distance,  #
            output_width=fixed_image_size[1],  #
            output_height=fixed_image_size[0],  #
            output_offset=fixed_image_offset,  #
            detector_spacing=fixed_image_spacing,  #
        ),  #
    }


@dadg_updater(names_returned=["of_values"])
def apply_sim_metric(  #
        *,  #
        sim_metric: str,  #
        moving_images: Float32[torch.Tensor, "b n m"],  #
        fixed_images: Float32[torch.Tensor, "#b n m"],  #
        weight_images: Float32[torch.Tensor, "#b n m"],  #
) -> dict[str, Any]:
    p_sim_met: ParametrisedSimilarityMetric = string_to_sim_met(sim_metric)
    return {  #
        "of_values": -p_sim_met.func_weighted(moving_images, fixed_images, weight_images, dim=(-1, -2)),  #
    }
