from typing import Any, Literal

import torch
from jaxtyping import Float32, Float64

import reg23_core
from reg23_experiments.data import sinogram
from reg23_experiments.data.structs import SceneGeometry, Transformation
from reg23_experiments.experiments.helpers import string_to_sim_met
from reg23_experiments.ops.data_manager import dadg_updater
from reg23_experiments.ops.geometry import get_crop_full_depth_drr, get_crop_nonzero_drr
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation

__all__ = ["refresh_scaling_images", "refresh_weights", "project_moving_images", "apply_sim_metric", "refresh_cropping",
           "resample_for_moving_image_grangeat", "apply_sim_metric_grangeat"]


# @dadg_updater(names_returned=["scaling_images", "fixed_image"])
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
    fixed_image = cropped_target.unsqueeze(0)
    return {  #
        "scaling_images": scaling_images,  #
        "fixed_image": fixed_image,  #
    }


def refresh_weights(  #
        *,  #
        weighting_method: Literal["none", "linear", "smooth_step", "gaussian"],  #
        weight_alpha: float,  #
        scaling_images: Float32[torch.Tensor, "b n m"],  #
        weight_epsilon: float = 1e-5,  #
) -> dict[str, Any]:
    if weighting_method == "linear":
        weight_images = scaling_images.clone()
    elif weighting_method == "smooth_step":
        if weight_alpha < 1e-2:
            weight_images = scaling_images.clone()
            weight_images[weight_images < 1.0 - weight_epsilon] = 0.0
        else:
            sc_sq = scaling_images.square()
            weight_images = torch.pow(3.0 * sc_sq - 2.0 * scaling_images * sc_sq, 1.0 / (weight_alpha * weight_alpha))
    elif weighting_method == "gaussian":
        if weight_alpha < 1e-2:
            weight_images = scaling_images.clone()
            weight_images[weight_images < 1.0 - weight_epsilon] = 0.0
        else:
            weight_images = (-((scaling_images - 1.0) / (0.6 * weight_alpha)).square()).exp()
            weight_images[scaling_images < weight_epsilon] = 0.0
    else:
        # weighting_method == "none"
        weight_images = None
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
        fixed_image: Float32[torch.Tensor, "n m"],  #
        weight_images: Float32[torch.Tensor, "#b n m"] | None,  #
) -> dict[str, Any]:
    return {  #
        "of_values": -string_to_sim_met(sim_metric)(  #
            fixed_image,  #
            moving_images,  #
            weights=weight_images,  #
        ),  #
    }


@dadg_updater(names_returned=["moving_images_grangeat"])
def resample_for_moving_image_grangeat(  #
        *,  #
        parameters: Float64[torch.Tensor, "b 6"],  #
        vif: sinogram.Sinogram,  #
        sinogram2d_grid: sinogram.Sinogram2dGrid,  #
        source_distance: float,  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
) -> dict[str, Any]:
    device = vif.device
    scene_geometry = SceneGeometry(source_distance=source_distance, fixed_image_offset=fixed_image_offset)
    p_matrix = SceneGeometry.projection_matrix(source_position=scene_geometry.source_position(device=device))

    resampleds = torch.empty((parameters.size()[0], *sinogram2d_grid.phi.size()))
    for i, p in enumerate(parameters):
        ph_matrix: torch.Tensor = torch.matmul(  #
            p_matrix,  #
            mapping_parameters_to_transformation(p).with_translation_offset(translation_offset).get_h(device=device)  #
        ).to(dtype=torch.float32)

        resampleds[i] = vif.resample_cuda_texture(ph_matrix, sinogram2d_grid)
    return {"moving_images_grangeat": resampleds}


@dadg_updater(names_returned=["of_values_grangeat"])
def apply_sim_metric_grangeat(  #
        *,  #
        sim_metric: str,  #
        moving_images_grangeat: torch.Tensor,  #
        sinogram2d: torch.Tensor,  #
) -> dict[str, Any]:
    return {  #
        "of_values_grangeat": -string_to_sim_met(sim_metric)(moving_images_grangeat, sinogram2d.unsqueeze(0)),  #
    }
