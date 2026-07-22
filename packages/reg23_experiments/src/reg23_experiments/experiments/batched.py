from typing import Callable

import torch
from jaxtyping import Float64

import reg23_core
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation

__all__ = ["objective_function_binary_weighted"]


def objective_function_binary_weighted(  #
        weighted_sim_metric: Callable,  #
        parameters: Float64[torch.Tensor, "b 6"],  #
        *,  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        current_transformation: Transformation,  #
        fixed_image_size: torch.Size,  #
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
        t.inverse().get_h(device=device)  #
        for t in ts], dim=0)
    # Projecting the mask
    masks = reg23_core.project_drr_cuboid_masks_batched(  #
        volume_size=torch.tensor(ct_volumes[0].size(), device=device).flip(dims=(0,)),  #
        voxel_spacing=ct_spacing,  #
        inverse_h_matrices=h_invs,  #
        source_distance=source_distance,  #
        output_width=fixed_image_size[1],  #
        output_height=fixed_image_size[0],  #
        output_offset=fixed_image_offset,  #
        detector_spacing=fixed_image_spacing  #
    )
    # Getting the resulting moving and fixed images
    moving_image: torch.Tensor | Error = data_manager().get("moving_image")
    fixed_image: torch.Tensor | Error = data_manager().get("fixed_image")
    # Comparing, potentially weighting with a mask
    if apply_mask and weight_with_mask:
        mask: torch.Tensor | Error = data_manager().get("mask")
        return -p_sim_met.func_weighted(moving_image, fixed_image, mask)
    return -p_sim_met.func(moving_image, fixed_image)
