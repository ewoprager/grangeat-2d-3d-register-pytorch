import torch


def radon2d(image: torch.Tensor, x_spacing: float, y_spacing: float, output_height: int, output_width: int,
            samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d.default(image, x_spacing, y_spacing, output_height, output_width,
                                                   samples_per_line)


def radon2d_v2(image: torch.Tensor, x_spacing: float, y_spacing: float, output_height: int, output_width: int,
               samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d_v2.default(image, x_spacing, y_spacing, output_height, output_width,
                                                      samples_per_line)


def dRadon2dDR(image: torch.Tensor, x_spacing: float, y_spacing: float, output_height: int, output_width: int,
               samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.dRadon2dDR.default(image, x_spacing, y_spacing, output_height, output_width,
                                                      samples_per_line)


def radon3d(volume: torch.Tensor, x_spacing: float, y_spacing: float, z_spacing: float, output_depth: int,
            output_height: int, output_width: int, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon3d.default(volume, x_spacing, y_spacing, z_spacing, output_depth, output_height,
                                                   output_width, samples_per_direction)


def radon3d_v2(volume: torch.Tensor, x_spacing: float, y_spacing: float, z_spacing: float, output_depth: int,
               output_height: int, output_width: int, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon3d_v2.default(volume, x_spacing, y_spacing, z_spacing, output_depth,
                                                      output_height, output_width, samples_per_direction)
