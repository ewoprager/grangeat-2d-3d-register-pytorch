import torch


def mymuladd(a: torch.Tensor, b: torch.Tensor, c: float) -> torch.Tensor:
    return torch.ops.ExtensionTest.mymuladd.default(a, b, c)


def radon2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.ops.ExtensionTest.radon2d.default(a, b)