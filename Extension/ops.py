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


def d_radon2d_dr(image: torch.Tensor, image_spacing: torch.Tensor, phi_values: torch.Tensor, r_values: torch.Tensor,
                 samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.d_radon2d_dr.default(image, image_spacing, phi_values, r_values, samples_per_line)


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


def d_radon3d_dr(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor,
                 theta_values: torch.Tensor, r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.d_radon3d_dr.default(volume, volume_spacing, phi_values, theta_values, r_values,
                                                        samples_per_direction)


def d_radon3d_dr_v2(volume: torch.Tensor, volume_spacing: torch.Tensor, phi_values: torch.Tensor,
                    theta_values: torch.Tensor, r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.d_radon3d_dr_v2.default(volume, volume_spacing, phi_values, theta_values, r_values,
                                                           samples_per_direction)


def resample_sinogram3d(sinogram3d: torch.Tensor, sinogram_spacing: torch.Tensor, sinogram_range_centres: torch.Tensor,
                        projection_matrix: torch.Tensor, phi_values: torch.Tensor,
                        r_values: torch.Tensor) -> torch.Tensor:
    return torch.ops.ExtensionTest.resample_sinogram3d.default(sinogram3d, sinogram_spacing, sinogram_range_centres,
                                                               projection_matrix, phi_values, r_values)


def normalised_cross_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.ExtensionTest.normalised_cross_correlation.default(a, b)


def grid_sample3d(input_: torch.Tensor, grid: torch.Tensor, address_mode: str="zero") -> torch.Tensor:
    return torch.ops.ExtensionTest.grid_sample3d.default(input_, grid.to(dtype=torch.float64), address_mode)
