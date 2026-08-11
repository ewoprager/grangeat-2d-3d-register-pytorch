from typing import Any, Literal

import torch
from jaxtyping import Float32, Float64

import reg23_core
from reg23_experiments.data.structs import Transformation
from reg23_experiments.experiments.helpers import ParametrisedSimilarityMetric, string_to_sim_met
from reg23_experiments.ops.data_manager import dadg_updater
from reg23_experiments.ops.geometry import get_crop_full_depth_drr, get_crop_nonzero_drr
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation

__all__ = ["refresh_scaling_images", "refresh_weights", "project_moving_images", "apply_sim_metric", "refresh_cropping"]


@dadg_updater(names_returned=["scaling_images", "fixed_images"])
def refresh_scaling_images(  #
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
        apply_scaling: bool,  #
) -> dict[str, Any]:
    ts: list[Transformation] = [mapping_parameters_to_transformation(p) for p in parameters]
    h_invs: torch.Tensor = torch.stack([  #
        t.with_translation_offset(translation_offset).inverse().get_h(device=ct_volumes[0].device)  #
        for t in ts  #
    ], dim=0)
    scaling_images = reg23_core.project_drr_cuboid_masks_batched(  #
        volume_size=torch.tensor(ct_volumes[0].size(), device=ct_volumes[0].device).flip(dims=(0,)),  #
        voxel_spacing=ct_spacing,  #
        inverse_h_matrices=h_invs,  #
        source_distance=source_distance,  #
        output_width=fixed_image_size[1],  #
        output_height=fixed_image_size[0],  #
        output_offset=fixed_image_offset,  #
        detector_spacing=fixed_image_spacing  #
    )
    # Generate the fixed images
    if apply_scaling:
        fixed_images = scaling_images * cropped_target.unsqueeze(0)
    else:
        fixed_images = cropped_target.unsqueeze(0)
    return {  #
        "scaling_images": scaling_images,  #
        "fixed_images": fixed_images,  #
    }


def refresh_weights(  #
        *,  #
        weighting: None | Literal["linear"] | float,  #
        scaling_images: Float32[torch.Tensor, "b n m"],  #
        weight_epsilon: float = 1e-5,  #
) -> dict[str, Any]:
    if weighting is None:
        weight_images = None
    elif isinstance(weighting, float):
        if weighting < 1e-2:
            weight_images = scaling_images.clone()
            weight_images[weight_images < 1.0 - weight_epsilon] = 0.0
        else:
            sc_sq = scaling_images.square()
            weight_images = torch.pow(3.0 * sc_sq - 2.0 * scaling_images * sc_sq, 1.0 / (weighting * weighting))
    else:
        assert weighting == "linear"
        weight_images = scaling_images.clone()
    return {  #
        "weight_images": weight_images,  #
    }


def refresh_cropping(  #
        *,  #
        parameters: Float64[torch.Tensor, "1 6"],  #
        cropping_method: Literal["none", "bounding_box", "valid_only"],  #
        image_2d_full: Float32[torch.Tensor, "n m"],  #
        source_distance: float,  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        image_2d_full_spacing: Float64[torch.Tensor, "2"],  #
        translation_offset: Float64[torch.Tensor, "2"],  #
) -> dict[str, Any]:
    """
    !Requires a batch size of 1!
    :param parameters:
    :param cropping_method:
    :param image_2d_full:
    :param source_distance:
    :param ct_volumes:
    :param ct_spacing:
    :param image_2d_full_spacing:
    :param translation_offset:
    :return:
    """
    current_transformation = mapping_parameters_to_transformation(parameters[0])
    if cropping_method == "none":
        cropping = None
    elif cropping_method == "bounding_box":
        cropping = get_crop_nonzero_drr(  #
            image_2d_full=image_2d_full,  #
            source_distance=source_distance,  #
            ct_volumes=ct_volumes,  #
            current_transformation=current_transformation,  #
            ct_spacing=ct_spacing,  #
            image_2d_full_spacing=image_2d_full_spacing,  #
            translation_offset=translation_offset,  #
        )
    else:
        assert cropping_method == "valid_only"
        cropping = get_crop_full_depth_drr(  #
            image_2d_full=image_2d_full,  #
            source_distance=source_distance,  #
            ct_volumes=ct_volumes,  #
            current_transformation=current_transformation,  #
            ct_spacing=ct_spacing,  #
            image_2d_full_spacing=image_2d_full_spacing,  #
            translation_offset=translation_offset,  #
        )
    return {"further_cropping": cropping}


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
        weight_images: Float32[torch.Tensor, "#b n m"] | None,  #
) -> dict[str, Any]:
    p_sim_met: ParametrisedSimilarityMetric = string_to_sim_met(sim_metric)
    if weight_images is None:
        sim = p_sim_met.func(moving_images, fixed_images, dim=(-1, -2))
    else:
        sim = p_sim_met.func_weighted(moving_images, fixed_images, weight_images, dim=(-1, -2))
    return {  #
        "of_values": -sim,  #
    }
