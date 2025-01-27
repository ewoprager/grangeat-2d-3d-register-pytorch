import torch


def radon2d(image: torch.Tensor, x_spacing: float, y_spacing: float, phi_values: torch.Tensor, r_values: torch.Tensor,
            samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d.default(image, x_spacing, y_spacing, phi_values, r_values, samples_per_line)


def radon2d_v2(image: torch.Tensor, x_spacing: float, y_spacing: float, phi_values: torch.Tensor,
               r_values: torch.Tensor, samples_per_line: int) -> torch.Tensor:
    """
    Note: CPU implementation is identical to that of `radon2d`
    """
    return torch.ops.ExtensionTest.radon2d_v2.default(image, x_spacing, y_spacing, phi_values, r_values,
                                                      samples_per_line)


def dRadon2dDR(image: torch.Tensor, x_spacing: float, y_spacing: float, phi_values: torch.Tensor,
               r_values: torch.Tensor, samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.dRadon2dDR.default(image, x_spacing, y_spacing, phi_values, r_values,
                                                      samples_per_line)


def radon3d(volume: torch.Tensor, x_spacing: float, y_spacing: float, z_spacing: float, phi_values: torch.Tensor,
            theta_values: torch.Tensor, r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon3d.default(volume, x_spacing, y_spacing, z_spacing, phi_values, theta_values,
                                                   r_values, samples_per_direction)


def radon3d_v2(volume: torch.Tensor, x_spacing: float, y_spacing: float, z_spacing: float, phi_values: torch.Tensor,
               theta_values: torch.Tensor, r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    """
    Note: CPU implementation is identical to that of `radon3d`
    """
    return torch.ops.ExtensionTest.radon3d_v2.default(volume, x_spacing, y_spacing, z_spacing, phi_values, theta_values,
                                                      r_values, samples_per_direction)


def dRadon3dDR(volume: torch.Tensor, x_spacing: float, y_spacing: float, z_spacing: float, phi_values: torch.Tensor,
               theta_values: torch.Tensor, r_values: torch.Tensor, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.dRadon3dDR.default(volume, x_spacing, y_spacing, z_spacing, phi_values, theta_values,
                                                      r_values, samples_per_direction)
