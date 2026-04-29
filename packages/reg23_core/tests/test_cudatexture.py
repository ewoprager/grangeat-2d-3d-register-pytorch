import pytest
import weakref
import gc

import torch

if torch.cuda.is_available():
    from reg23 import CUDATexture2D, CUDATexture3D
    from reg23 import resample_sinogram3d_cuda_texture


    def test_cuda_texture():
        device = torch.device("cuda")

        d2 = torch.tensor([[1, 2], [3, 4]], device=device, dtype=torch.float32)

        CUDATexture2D(d2, "zero", "zero")

        d3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device, dtype=torch.float32)
        CUDATexture3D(d3, "zero", "zero", "zero")

        with pytest.raises(TypeError):
            CUDATexture3D(d3, "zero", "zero")

        with pytest.raises(RuntimeError):
            CUDATexture3D(d2, "zero", "zero", "zero")

        with pytest.raises(RuntimeError):
            CUDATexture3D(d3.to(device=torch.device("cpu")), "zero", "zero", "zero")


    def test_resampling():
        device = torch.device("cuda")

        d3 = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], device=device, dtype=torch.float32)
        texture = CUDATexture3D(d3, "zero", "zero", "zero")
        res = resample_sinogram3d_cuda_texture(texture, "classic", 0.1, torch.eye(4, device=device),
            torch.zeros(1, device=device), torch.zeros(1, device=device))


    def test_cyclic_reference():
        tensor = torch.zeros(3, 3, 3, device=torch.device("cuda"))
        texture = CUDATexture3D(tensor, "zero", "zero", "zero")
        ref = weakref.ref(texture)
        assert ref() is not None
        del texture
        gc.collect()
        assert ref() is None
