import torch

from . import ops

__all__ = []


def project_drr_setup_context(ctx, inputs, output):
    (volume, voxel_spacing, homography_matrix_inverse, source_distance, output_width, output_height, output_offset,
     detector_spacing) = inputs
    ctx.save_for_backward(volume, voxel_spacing, homography_matrix_inverse, output_offset, detector_spacing)
    ctx.source_distance = source_distance
    ctx.output_width = output_width
    ctx.output_height = output_height


def project_drr_backward(ctx, d_loss_d_drr):
    """
    :param ctx:
    :param d_loss_d_drr: tensor of size (drr_height, drr_width); the derivative of the loss w.r.t. the DRR
    :return: tensor of size (4, 4): the derivative of the loss w.r.t. the inverse homography matrix
    """
    (volume, voxel_spacing, homography_matrix_inverse, output_offset, detector_spacing) = ctx.saved_tensors
    source_distance = ctx.source_distance
    output_width = ctx.output_width
    output_height = ctx.output_height

    d_loss_d_hmi = ops.project_drr_backward(volume, voxel_spacing, homography_matrix_inverse, source_distance,
                                            output_width, output_height, output_offset, detector_spacing, d_loss_d_drr)

    return None, None, d_loss_d_hmi, None, None, None, None, None, None


torch.library.register_autograd("reg23::project_drr", project_drr_backward, setup_context=project_drr_setup_context)


def normalised_cross_correlation_setup_context(ctx, inputs, output):
    (a, b) = inputs
    (_, sum_a, sum_b, numerator, denominator_left, denominator_right) = output
    ctx.save_for_backward(a, b)
    ctx.numerator = numerator
    ctx.sum_a = sum_a
    ctx.sum_b = sum_b
    ctx.denominator_left = denominator_left
    ctx.denominator_right = denominator_right


def normalised_cross_correlation_backward(ctx, d_loss_d_zncc, *args):
    """
    :param ctx:
    :param d_loss_d_zncc: tensor of size (,); the derivative of the loss w.r.t. the ZNCC
    :return: tensor of the same size as a: the derivative of the loss w.r.t. the tensor `a`
    """
    (a, b) = ctx.saved_tensors

    n_f = float(a.numel())
    denominator_left_squared = ctx.denominator_left * ctx.denominator_left
    d_zncc_d_a = ((denominator_left_squared * (n_f * b - ctx.sum_b) - ctx.numerator * (n_f * a - ctx.sum_a)) /  #
                  (denominator_left_squared * ctx.denominator_left * ctx.denominator_right))

    return d_loss_d_zncc * d_zncc_d_a, None


torch.library.register_autograd("reg23::normalised_cross_correlation", normalised_cross_correlation_backward,
                                setup_context=normalised_cross_correlation_setup_context)
