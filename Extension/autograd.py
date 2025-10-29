import torch

from .autograd_impl import *


def project_drr(homography_matrix_inverse: torch.Tensor, volume: torch.Tensor, voxel_spacing: torch.Tensor,
                source_distance: float, output_width: int, output_height: int, output_offset: torch.Tensor,
                detector_spacing: torch.Tensor) -> torch.Tensor:
    return ProjectDRR.apply(homography_matrix_inverse, volume, voxel_spacing, source_distance, output_width,
                            output_height, output_offset, detector_spacing)
