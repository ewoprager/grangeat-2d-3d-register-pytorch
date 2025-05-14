import pytest
import torch

from Extension import project_drr

def test_project_drr():
    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    h_matrix_inv = torch.eye(4)
    source_distance = 1000.0
    output_size = torch.Size([10, 15])
    detector_spacing = torch.tensor([0.2, 0.25])
    res = project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0], detector_spacing)
    assert res.size() == output_size
    if torch.cuda.is_available():
        res_cuda = project_drr(input_.cuda(), voxel_spacing.cuda(), h_matrix_inv.cuda(), source_distance, output_size[1], output_size[0], detector_spacing.cuda())
        assert res == pytest.approx(res_cuda.cpu(), abs=0.01)

    # input must be 3D, so these should raise a runtime error
    input_ = torch.rand((11, 12, 8, 5))
    with pytest.raises(RuntimeError):
        project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0], detector_spacing)
    input_ = torch.rand((11, 12))
    with pytest.raises(RuntimeError):
        project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0], detector_spacing)

    # voxel spacing and detector spacing must be 1D and of lengths 3 & 2 respectively, so these should raise a runtime
    # error
    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.2, 0.1])
    with pytest.raises(RuntimeError):
        project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0], detector_spacing)
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    detector_spacing = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(RuntimeError):
        project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0], detector_spacing)
