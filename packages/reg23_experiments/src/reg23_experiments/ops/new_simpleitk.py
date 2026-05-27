import SimpleITK as sitk
import kornia
import torch


def transform_to_parameters(transform: sitk.Euler3DTransform) -> torch.Tensor:
    return kornia.geometry.conversions.rotation_matrix_to_axis_angle(
        torch.tensor(transform.GetMatrix()).reshape((3, 3)).unsqueeze(0))[0]


def parameter_to_transform(parameters: torch.Tensor) -> sitk.Euler3DTransform:
    pass
