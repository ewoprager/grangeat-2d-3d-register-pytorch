import kornia
import SimpleITK as sitk
import torch
from jaxtyping import Float64

__all__ = ["transform_to_parameters", "parameter_to_transform"]

_PARAMETER_SCALES = torch.tensor([32.0, 32.0, 32.0, 1.0, 1.0, 0.05])
_PARAMETER_SCALES_INV = 1.0 / _PARAMETER_SCALES


def transform_to_parameters(transform: sitk.Euler3DTransform, *, device: torch.device = torch.device("cpu")) -> Float64[
    torch.Tensor, "6"]:
    rotation = kornia.geometry.conversions.rotation_matrix_to_axis_angle(
        torch.tensor(transform.GetMatrix(), dtype=torch.float64, device=device).reshape((3, 3)).unsqueeze(0))[0]
    translation = torch.tensor(transform.GetTranslation(), dtype=torch.float64, device=device)
    return torch.cat((rotation, translation)) * _PARAMETER_SCALES


def parameter_to_transform(parameters: Float64[torch.Tensor, "6"]) -> sitk.Euler3DTransform:
    unscaled = parameters.cpu() * _PARAMETER_SCALES_INV
    rotation = unscaled[0:3]
    translation = unscaled[3:6]
    ret = sitk.Euler3DTransform()
    ret.SetMatrix(kornia.geometry.conversions.axis_angle_to_rotation_matrix(rotation.unsqueeze(0))[0].numpy())
    ret.SetTranslation(translation.numpy())
    return ret


def parameters_to_affine_matrix(parameters: Float64[torch.Tensor, "6"]) -> Float64[torch.Tensor, "4 4"]:
    unscaled = parameters * _PARAMETER_SCALES_INV
    rotation = unscaled[0:3]
    translation = unscaled[3:6]
    r = kornia.geometry.conversions.axis_angle_to_rotation_matrix(rotation.unsqueeze(0))[0]
    rt = torch.hstack([r, translation.t().unsqueeze(-1)])
    return torch.vstack(
        [rt, torch.tensor([0., 0., 0., 1.], device=parameters.device, dtype=torch.float64).unsqueeze(0)])
