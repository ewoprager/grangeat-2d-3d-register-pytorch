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
        assert res == pytest.approx(res_cuda.cpu())

    input_ = torch.rand((11, 12, 8, 5))
    with pytest.raises(RuntimeError):
        grid_sample3d(input_, grid)
    input_ = torch.rand((11, 12))
    with pytest.raises(RuntimeError):
        grid_sample3d(input_, grid)

    input_ = torch.rand((11, 12, 8))
    grid = torch.rand((10, 7, 2))
    with pytest.raises(RuntimeError):
        grid_sample3d(input_, grid)
    grid = torch.rand((10, 7, 4))
    with pytest.raises(RuntimeError):
        grid_sample3d(input_, grid)
