import pytest

import torch

from Extension import reg23


def test_cuda_texture():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    d2 = torch.tensor([[1, 2], [3, 4]], device=device, dtype=torch.float32)
    reg23.CUDATexture2D(d2, "zero", "zero")

    device = torch.device("cuda")
    d3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device, dtype=torch.float32)
    reg23.CUDATexture3D(d3, "zero", "zero", "zero")

    with pytest.raises(TypeError):
        reg23.CUDATexture3D(d3, "zero", "zero")

    with pytest.raises(RuntimeError):
        reg23.CUDATexture3D(d2, "zero", "zero", "zero")

    with pytest.raises(RuntimeError):
        reg23.CUDATexture3D(d3.to(device=torch.device("cpu")), "zero", "zero", "zero")


