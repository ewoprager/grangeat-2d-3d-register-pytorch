import SimpleITK as sitk
import kornia
import torch

__all__ = ["transform_to_parameters", "parameter_to_transform"]

_PARAMETER_SCALES = torch.tensor([32.0, 32.0, 32.0, 1.0, 1.0, 0.05])
_PARAMETER_SCALES_INV = 1.0 / _PARAMETER_SCALES


def transform_to_parameters(transform: sitk.Euler3DTransform, **tensor_kwargs) -> torch.Tensor:
    rotation = kornia.geometry.conversions.rotation_matrix_to_axis_angle(
        torch.tensor(transform.GetMatrix(), **tensor_kwargs).reshape((3, 3)).unsqueeze(0))[0]
    translation = torch.tensor(transform.GetTranslation(), **tensor_kwargs)
    return torch.cat((rotation, translation)) * _PARAMETER_SCALES


def parameter_to_transform(parameters: torch.Tensor) -> sitk.Euler3DTransform:
    scaled = parameters.cpu() * _PARAMETER_SCALES_INV
    rotation = scaled[0:3]
    translation = scaled[3:6]
    ret = sitk.Euler3DTransform()
    ret.SetMatrix(kornia.geometry.conversions.axis_angle_to_rotation_matrix(rotation).numpy())
    ret.SetTranslation(translation.numpy())
    return ret
