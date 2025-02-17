import torch


def radon2d(image: torch.Tensor, image_spacing: torch.Tensor, phi_values: torch.Tensor, r_values: torch.Tensor,
            samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d.default(image, image_spacing, phi_values, r_values, samples_per_line)


def radon2d_v2(image: torch.Tensor, image_spacing: torch.Tensor, phi_values: torch.Tensor, r_values: torch.Tensor,
               samples_per_line: int) -> torch.Tensor:
    """
    Note: CPU implementation is identical to that of `radon2d`
    """
    return torch.ops.ExtensionTest.radon2d_v2.default(image, image_spacing, phi_values, r_values, samples_per_line)


def dRadon2dDR(image: torch.Tensor, image_spacing: torch.Tensor, phi_values: torch.Tensor, r_values: torch.Tensor,
               samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.dRadon2dDR.default(image, image_spacing, phi_values, r_values, samples_per_line)


def radon3d(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor, theta_values: torch.Tensor,
            r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon3d.default(volume, volume_spacing, phi_values, theta_values, r_values,
                                                   samples_per_direction)


def radon3d_v2(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor, theta_values: torch.Tensor,
               r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    """
    Note: CPU implementation is identical to that of `radon3d`
    """
    return torch.ops.ExtensionTest.radon3d_v2.default(volume, volume_spacing, phi_values, theta_values, r_values,
                                                      samples_per_direction)


def dRadon3dDR(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor, theta_values: torch.Tensor,
               r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.dRadon3dDR.default(volume, volume_spacing, phi_values, theta_values, r_values,
                                                      samples_per_direction)


def dRadon3dDR_v2(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor,
                  theta_values: torch.Tensor, r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.dRadon3dDR_v2.default(volume, volume_spacing, phi_values, theta_values, r_values,
                                                         samples_per_direction)


def resample_radon_volume(sinogram3d: torch.Tensor, sinogram_spacing: torch.Tensor,
                          sinogram_range_centres: torch.Tensor, projection_matrix: torch.Tensor, phi_grid: torch.Tensor,
                          r_grid: torch.Tensor) -> torch.Tensor:
    return torch.ops.ExtensionTest.resample_radon_volume.default(sinogram3d, sinogram_spacing, sinogram_range_centres,
                                                                 projection_matrix, phi_grid, r_grid)
