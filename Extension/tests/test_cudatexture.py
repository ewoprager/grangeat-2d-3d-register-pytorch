import pytest

import torch

if torch.cuda.is_available():

    from Extension import reg23
    from Extension import resample_sinogram3d_cuda_texture


    def test_cuda_texture():
        device = torch.device("cuda")

        d2 = torch.tensor([[1, 2], [3, 4]], device=device, dtype=torch.float32)

        reg23.CUDATexture2D(d2, "zero", "zero")

        d3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device, dtype=torch.float32)
        reg23.CUDATexture3D(d3, "zero", "zero", "zero")

        with pytest.raises(TypeError):
            reg23.CUDATexture3D(d3, "zero", "zero")

        with pytest.raises(RuntimeError):
            reg23.CUDATexture3D(d2, "zero", "zero", "zero")

        with pytest.raises(RuntimeError):
            reg23.CUDATexture3D(d3.to(device=torch.device("cpu")), "zero", "zero", "zero")


    def test_resampling():
        device = torch.device("cuda")

        d3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device, dtype=torch.float32)
        texture = reg23.CUDATexture3D(d3, "zero", "zero", "zero")
        res = resample_sinogram3d_cuda_texture(
            texture, "classic", 0.1, torch.eye(4, device=device), torch.zeros(1, device=device),
            torch.zeros(1, device=device))
