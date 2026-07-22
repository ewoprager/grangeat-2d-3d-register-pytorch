from typing import Callable

import torch
from jaxtyping import Float32, Float64

import reg23_core
from reg23_experiments.data.structs import Transformation
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation

__all__ = ["objective_function_binary_weighted", "objective_function_alpha_weighted"]


def objective_function_binary_weighted(  #
        *,  #
        weighted_sim_metric: Callable,  #
        parameters: Float64[torch.Tensor, "b 6"],  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        fixed_image: Float32[torch.Tensor, "n m"],  #
        source_distance: float,  #
        fixed_image_spacing: Float64[torch.Tensor, "2"],  #
        downsample_level: int,  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
        weight_epsilon: float = 1e-5,  #
) -> torch.Tensor:
    device = parameters.device
    ts: list[Transformation] = [mapping_parameters_to_transformation(p) for p in parameters]
    h_invs: torch.Tensor = torch.stack([  #
        t.with_translation_offset(translation_offset).inverse().get_h(device=device)  #
        for t in ts], dim=0)
    # Project the weight images
    weights = reg23_core.project_drr_cuboid_masks_batched(  #
        volume_size=torch.tensor(ct_volumes[0].size(), device=device).flip(dims=(0,)),  #
        voxel_spacing=ct_spacing,  #
        inverse_h_matrices=h_invs,  #
        source_distance=source_distance,  #
        output_width=fixed_image.size()[1],  #
        output_height=fixed_image.size()[0],  #
        output_offset=fixed_image_offset,  #
        detector_spacing=fixed_image_spacing  #
    )
    weights[weights < 1.0 - weight_epsilon] = 0.0
    # Project the DRRs
    moving_images = reg23_core.project_drrs_batched(  #
        volume=ct_volumes[downsample_level],  #
        voxel_spacing=ct_spacing * 2.0 ** downsample_level,  #
        inverse_h_matrices=h_invs,  #
        source_distance=source_distance,  #
        output_width=fixed_image.size()[1],  #
        output_height=fixed_image.size()[0],  #
        output_offset=fixed_image_offset,  #
        detector_spacing=fixed_image_spacing,  #
    )
    return -weighted_sim_metric(  #
        moving_images,  #
        fixed_image.expand(moving_images.size()),  #
        weights,  #
        dim=(-1, -2)  #
    )


def objective_function_alpha_weighted(  #
        *,  #
        weighted_sim_metric: Callable,  #
        parameters: Float64[torch.Tensor, "b 6"],  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        fixed_image: Float32[torch.Tensor, "n m"],  #
        source_distance: float,  #
        fixed_image_spacing: Float64[torch.Tensor, "2"],  #
        downsample_level: int,  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
        weight_alpha: float,  #
) -> torch.Tensor:
    device = parameters.device
    ts: list[Transformation] = [mapping_parameters_to_transformation(p) for p in parameters]
    h_invs: torch.Tensor = torch.stack([  #
        t.with_translation_offset(translation_offset).inverse().get_h(device=device)  #
        for t in ts], dim=0)
    # Project the masks
    masks = reg23_core.project_drr_cuboid_masks_batched(  #
        volume_size=torch.tensor(ct_volumes[0].size(), device=device).flip(dims=(0,)),  #
        voxel_spacing=ct_spacing,  #
        inverse_h_matrices=h_invs,  #
        source_distance=source_distance,  #
        output_width=fixed_image.size()[1],  #
        output_height=fixed_image.size()[0],  #
        output_offset=fixed_image_offset,  #
        detector_spacing=fixed_image_spacing  #
    )
    # Generate the masked fixed images
    masked_fixed_images = masks * fixed_image.unsqueeze(0)
    # Calculate the weight images
    weights = torch.pow(3.0 * masks * masks - 2 * masks * masks * masks, 1.0 / weight_alpha)
    # Project the DRRs
    moving_images = reg23_core.project_drrs_batched(  #
        volume=ct_volumes[downsample_level],  #
        voxel_spacing=ct_spacing * 2.0 ** downsample_level,  #
        inverse_h_matrices=h_invs,  #
        source_distance=source_distance,  #
        output_width=fixed_image.size()[1],  #
        output_height=fixed_image.size()[0],  #
        output_offset=fixed_image_offset,  #
        detector_spacing=fixed_image_spacing,  #
    )
    return -weighted_sim_metric(  #
        moving_images,  #
        masked_fixed_images,  #
        weights,  #
        dim=(-1, -2)  #
    )
