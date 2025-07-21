import pytest
import torch

from Extension import grid_sample3d


def test_grid_sample3d():
    input_ = torch.rand((11, 12, 8))
    grid = torch.rand((10, 7, 3))
    res = grid_sample3d(input_, grid)
    assert res.size() == grid.size()[0:-1]
    if torch.cuda.is_available():
        res_cuda = grid_sample3d(input_.cuda(), grid.cuda())
        assert res == pytest.approx(res_cuda.cpu(), abs=0.01)

    # input must be 3D, so these should raise a runtime error
    input_ = torch.rand((11, 12, 8, 5))
    with pytest.raises(RuntimeError):
        grid_sample3d(input_, grid)
    input_ = torch.rand((11, 12))
    with pytest.raises(RuntimeError):
        grid_sample3d(input_, grid)

    # grid must have final dimension 3, so these should raise a runtime error
    input_ = torch.rand((11, 12, 8))
    grid = torch.rand((10, 7, 2))
    with pytest.raises(RuntimeError):
        grid_sample3d(input_, grid)
    grid = torch.rand((10, 7, 4))
    with pytest.raises(RuntimeError):
        grid_sample3d(input_, grid)


def test_grid_sample3d_against_torch():
    # on cpu:
    device = torch.device("cpu")
    texture = torch.tensor([[[1.0, 2.0], [4.0, 3.0]]], device=device)
    xs = torch.linspace(-2.5, 2.5, 50, device=device)
    ys = torch.linspace(-2.5, 2.5, 50, device=device)
    zs = torch.zeros(1, device=device)
    zs, ys, xs = torch.meshgrid(zs, ys, xs)
    grid = torch.stack((xs, ys, zs), dim=-1)
    res = grid_sample3d(texture, grid, "zero", "zero", "zero")
    res_torch = \
    torch.nn.functional.grid_sample(texture.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0), padding_mode="zeros")[0, 0]
    assert res == pytest.approx(res_torch, abs=1e-5)

    # on cuda:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        texture = texture.to(device=device)
        grid = grid.to(device=device)
        res = grid_sample3d(texture, grid, "zero", "zero", "zero")
        res_torch = \
        torch.nn.functional.grid_sample(texture.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0), padding_mode="zeros")[
            0, 0]
        assert res.cpu() == pytest.approx(res_torch.cpu(), abs=1e-5)
