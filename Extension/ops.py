import torch


def radon2d(a: torch.Tensor, height: int, width: int, samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d.default(a, height, width, samples_per_line)


def radon2d_v2(a: torch.Tensor, height: int, width: int, samples_per_line: int) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d_v2.default(a, height, width, samples_per_line)
