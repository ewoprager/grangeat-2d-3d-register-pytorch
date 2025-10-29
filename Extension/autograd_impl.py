import torch

from . import ops


class ProjectDRR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, homography_matrix_inverse: torch.Tensor, volume: torch.Tensor, voxel_spacing: torch.Tensor,
                source_distance: float, output_width: int, output_height: int, output_offset: torch.Tensor,
                detector_spacing: torch.Tensor) -> torch.Tensor:
        output = ops.project_drr(volume, voxel_spacing, homography_matrix_inverse, source_distance, output_width,
                                 output_height, output_offset, detector_spacing)
        ctx.save_for_backwards(homography_matrix_inverse, volume, voxel_spacing, source_distance, output_width,
                               output_height, output_offset, detector_spacing)
        return output

    @staticmethod
    def backward(ctx, d_loss_d_drr):
        """
        :param ctx:
        :param d_loss_d_drr: tensor of size (drr_height, drr_width); the derivative of the loss w.r.t. the DRR
        :return: tensor of size (
        """
        (homography_matrix_inverse, volume, voxel_spacing, source_distance, output_width, output_height, output_offset,
         detector_spacing) = ctx.saved_tensors

        d_drr_d_hmi = ops.d_project_drr_d_hmi(volume, voxel_spacing, homography_matrix_inverse, source_distance,
                                              output_width, output_height, output_offset, detector_spacing)

        d_loss_d_hmi = torch.einsum("ji,jikl->kl", d_loss_d_drr, d_drr_d_hmi)

        return d_loss_d_hmi, None, None, None, None, None, None, None