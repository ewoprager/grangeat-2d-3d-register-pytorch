import torch


def radon2d(a: torch.Tensor, x_spacing: float, y_spacing: float, height: int, width: int,
            samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d.default(a, x_spacing, y_spacing, height, width, samples_per_line)


def radon2d_v2(a: torch.Tensor, x_spacing: float, y_spacing: float, height: int, width: int,
               samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d_v2.default(a, x_spacing, y_spacing, height, width, samples_per_line)


def radon3d(a: torch.Tensor, x_spacing: float, y_spacing: float, z_spacing: float, depth: int, height: int, width: int,
            samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon3d.default(a, x_spacing, y_spacing, z_spacing, depth, height, width,
                                                   samples_per_direction)


def radon3d_v2(a: torch.Tensor, x_spacing: float, y_spacing: float, z_spacing: float, depth: int, height: int,
               width: int, samples_per_direction: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon3d_v2.default(a, x_spacing, y_spacing, z_spacing, depth, height, width,
                                                      samples_per_direction)
