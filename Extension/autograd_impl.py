import torch

from . import ops


class ProjectDRR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, homography_matrix_inverse: torch.Tensor, volume: torch.Tensor, voxel_spacing: torch.Tensor,
                source_distance: float, output_width: int, output_height: int, output_offset: torch.Tensor,
                detector_spacing: torch.Tensor) -> torch.Tensor:
        output = ops.project_drr(volume, voxel_spacing, homography_matrix_inverse, source_distance, output_width,
                                 output_height, output_offset, detector_spacing)
        ctx.save_for_backward(homography_matrix_inverse, volume, voxel_spacing, output_offset, detector_spacing)
        ctx.source_distance = source_distance
        ctx.output_width = output_width
        ctx.output_height = output_height
        return output

    @staticmethod
    def backward(ctx, d_loss_d_drr):
        """
        :param ctx:
        :param d_loss_d_drr: tensor of size (drr_height, drr_width); the derivative of the loss w.r.t. the DRR
        :return: tensor of size (4, 4): the derivative of the loss w.r.t. the inverse homography matrix
        """
        (homography_matrix_inverse, volume, voxel_spacing, output_offset, detector_spacing) = ctx.saved_tensors
        source_distance = ctx.source_distance
        output_width = ctx.output_width
        output_height = ctx.output_height

        d_loss_d_hmi = ops.project_drr_backward(volume, voxel_spacing, homography_matrix_inverse, source_distance,
                                                output_width, output_height, output_offset, detector_spacing,
                                                d_loss_d_drr)

        return d_loss_d_hmi, None, None, None, None, None, None, None


class NormalisedCrossCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        output, sum_a, sum_b, numerator, denominator_left, denominator_right = ops.normalised_cross_correlation_forward(
            a, b)
        ctx.save_for_backward(a, b)
        ctx.numerator = numerator
        ctx.sum_a = sum_a
        ctx.sum_b = sum_b
        ctx.denominator_left = denominator_left
        ctx.denominator_right = denominator_right
        return output

    @staticmethod
    def backward(ctx, d_loss_d_zncc):
        """
        :param ctx:
        :param d_loss_d_zncc: tensor of size (,); the derivative of the loss w.r.t. the ZNCC
        :return: tensor of the same size as a: the derivative of the loss w.r.t. the tensor `a`
        """
        (a, b) = ctx.saved_tensors

        n_f = float(a.numel())
        denominator_left_squared = ctx.denominator_left * ctx.denominator_left
        d_loss_d_a = ((denominator_left_squared * (n_f * b - ctx.sum_b) - ctx.numerator * (n_f * a - ctx.sum_a)) /  #
                      (denominator_left_squared * ctx.denominator_left * ctx.denominator_right))

        return d_loss_d_a, None
