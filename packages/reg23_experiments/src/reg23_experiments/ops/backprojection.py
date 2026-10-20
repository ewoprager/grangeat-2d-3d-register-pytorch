import torch

from reg23_experiments.data.segmentation import OrderedPoints2D
from reg23_experiments.data.structs import Transformation

__all__ = ["ordered_points_to_rays"]


def ordered_points_to_rays(  #
    points: OrderedPoints2D,  #
    transformation: Transformation,  #
    source_distance: float,  #
) -> torch.Tensor:
    """
    Each ray represented as row of a tensor, with 6 elements:
        | <- 3-component intersection point -> | <- 3-component unit direction -> |
    :param points:
    :param transformation:
    :param source_distance:
    :return:
    """
    device = torch.device("cpu")
    source_position = torch.tensor([0.0, 0.0, source_distance], dtype=torch.float64, device=device)
    input_vectors = points.data.to(device=device)
    inputs_homogeneous = torch.cat((  #
        input_vectors,  #
        torch.ones(input_vectors.size()[:-1], dtype=torch.float64, device=device).unsqueeze(-1),  #
    ), dim=-1)
    tr_ps_homogeneous = torch.einsum("ji,...i->...j", transformation.get_h(device), inputs_homogeneous)
    transformed_points = tr_ps_homogeneous[..., 0:3]
