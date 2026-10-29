import torch
from jaxtyping import Float

from reg23_experiments.data.structs import Transformation

__all__ = ["ordered_points_to_rays", "ray_sphere_intersection"]


def ordered_points_to_rays(  #
    *,  #
    points: Float[torch.Tensor, "n 2"],  #
    transformation: Transformation,  #
    source_position: Float[torch.Tensor, "n 3"],  #
) -> tuple[Float[torch.Tensor, "n 3"], Float[torch.Tensor, "n 3"]]:
    """
    :param points:
    :param transformation:
    :param source_position:
    :return: tuple of (tensor of 3-component intersection points, tensor of 3-component unit directions)
    """
    device = torch.device("cpu")
    h_inv = transformation.inverse().get_h(device)
    source_position_homogenous = torch.cat((  #
        source_position,  #
        torch.ones(1, dtype=torch.float64, device=device),  #
    ), dim=-1)
    intersection_point_homogenous = torch.einsum("ji,i->j", h_inv, source_position_homogenous)

    points_3d = torch.cat((  #
        points,  #
        torch.zeros(points.size()[:-1], dtype=torch.float64, device=device).unsqueeze(-1),  #
    ), dim=-1)
    pre_tr_unit_dirs = torch.nn.functional.normalize(points_3d - source_position.unsqueeze(0), dim=-1)
    pre_tr_unit_dirs_homogenous = torch.cat((  #
        pre_tr_unit_dirs,  #
        torch.zeros(pre_tr_unit_dirs.size()[:-1], dtype=torch.float64, device=device).unsqueeze(-1),  #
    ), dim=-1)
    unit_dirs_homogenous = torch.einsum("ji,...i->...j", h_inv, pre_tr_unit_dirs_homogenous)

    return (  #
        intersection_point_homogenous[0:3].unsqueeze(0).expand(unit_dirs_homogenous.size()[0], -1),  #
        unit_dirs_homogenous[:, 0:3],  #
    )


def ray_sphere_intersection(  #
    *,  #
    ray_point: Float[torch.Tensor, "3"],  #
    ray_unit_dir: Float[torch.Tensor, "3"],  #
    sphere_centre: Float[torch.Tensor, "3"],  #
    sphere_radius: float,  #
) -> Float[torch.Tensor, "n 3"]:
    """
    Formulae from here: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    :param ray_point:
    :param ray_unit_dir:
    :param sphere_centre:
    :param sphere_radius:
    :return:
    """
    o_minus_c = ray_point - sphere_centre
    d_middle = -torch.dot(ray_unit_dir, o_minus_c)
    nabla = d_middle.square() - (o_minus_c.square().sum() - sphere_radius * sphere_radius)
    if nabla < 0:
        return torch.empty((0, 3))
    if nabla < 1e-8:
        return (ray_point + d_middle * ray_unit_dir).unsqueeze(0)
    return ray_point.unsqueeze(0) + (
            d_middle + torch.tensor([1.0, -1.0]).unsqueeze(-1) * nabla.sqrt()) * ray_unit_dir.unsqueeze(0)
