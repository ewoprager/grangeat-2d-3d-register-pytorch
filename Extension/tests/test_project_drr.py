import pytest
import torch

from Extension import project_drr, project_drr_cuboid_mask, autograd


def test_project_drr():
    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    h_matrix_inv = torch.eye(4)
    source_distance = 1000.0
    output_size = torch.Size([10, 15])
    detector_spacing = torch.tensor([0.2, 0.25])
    res = project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                      torch.zeros(2, dtype=torch.float64), detector_spacing)
    assert res.size() == output_size
    if torch.cuda.is_available():
        res_cuda = project_drr(input_.cuda(), voxel_spacing.cuda(), h_matrix_inv.cuda(), source_distance,
                               output_size[1], output_size[0], torch.zeros(2, dtype=torch.float64),
                               detector_spacing.cuda())
        assert res == pytest.approx(res_cuda.cpu(), abs=0.01)

    # input must be 3D, so these should raise a runtime error
    input_ = torch.rand((11, 12, 8, 5))
    with pytest.raises(RuntimeError):
        project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                    torch.zeros(2, dtype=torch.float64), detector_spacing)
    input_ = torch.rand((11, 12))
    with pytest.raises(RuntimeError):
        project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                    torch.zeros(2, dtype=torch.float64), detector_spacing)

    # voxel spacing and detector spacing must be 1D and of lengths 3 & 2 respectively, so these should raise a runtime
    # error
    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.2, 0.1])
    with pytest.raises(RuntimeError):
        project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                    torch.zeros(2, dtype=torch.float64), detector_spacing)
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    detector_spacing = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(RuntimeError):
        project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                    torch.zeros(2, dtype=torch.float64), detector_spacing)

    # output offset should shift the image
    detector_spacing = torch.tensor([0.2, 0.25])
    zero = project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                       torch.zeros(2, dtype=torch.float64), detector_spacing)
    offset = project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                         3 * detector_spacing.to(dtype=torch.float64), detector_spacing)
    assert zero[3:, 3:] == pytest.approx(offset[:-3, :-3])


def test_project_drr_cuboid_mask():
    input_size = torch.tensor([8, 12, 11])
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    h_matrix_inv = torch.eye(4)
    source_distance = 1000.0
    output_size = torch.Size([10, 15])
    detector_spacing = torch.tensor([0.2, 0.25])
    res = project_drr_cuboid_mask(input_size, voxel_spacing, h_matrix_inv, source_distance, output_size[1],
                                  output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing)
    assert res.size() == output_size
    if torch.cuda.is_available():
        res_cuda = project_drr_cuboid_mask(input_size.cuda(), voxel_spacing.cuda(), h_matrix_inv.cuda(),
                                           source_distance, output_size[1], output_size[0],
                                           torch.zeros(2, dtype=torch.float64).cuda(), detector_spacing.cuda())
        assert res == pytest.approx(res_cuda.cpu(), abs=0.01)

    # input must be 3D, so these should raise a runtime error
    input_size = torch.tensor([11, 12, 8, 5])
    with pytest.raises(RuntimeError):
        project_drr_cuboid_mask(input_size, voxel_spacing, h_matrix_inv, source_distance, output_size[1],
                                output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing)
    input_size = torch.tensor([11, 12])
    with pytest.raises(RuntimeError):
        project_drr_cuboid_mask(input_size, voxel_spacing, h_matrix_inv, source_distance, output_size[1],
                                output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing)

    # voxel spacing and detector spacing must be 1D and of lengths 3 & 2 respectively, so these should raise a runtime
    # error
    input_size = torch.tensor([8, 12, 11])
    voxel_spacing = torch.tensor([0.2, 0.1])
    with pytest.raises(RuntimeError):
        project_drr_cuboid_mask(input_size, voxel_spacing, h_matrix_inv, source_distance, output_size[1],
                                output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing)
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    detector_spacing = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(RuntimeError):
        project_drr_cuboid_mask(input_size, voxel_spacing, h_matrix_inv, source_distance, output_size[1],
                                output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing)

    # output offset should shift the image
    detector_spacing = torch.tensor([0.2, 0.25])
    zero = project_drr_cuboid_mask(input_size, voxel_spacing, h_matrix_inv, source_distance, output_size[1],
                                   output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing)
    offset = project_drr_cuboid_mask(input_size, voxel_spacing, h_matrix_inv, source_distance, output_size[1],
                                     output_size[0], 3 * detector_spacing.to(dtype=torch.float64), detector_spacing)
    assert zero[3:, 3:] == pytest.approx(offset[:-3, :-3], abs=0.001)


import matplotlib.pyplot as plt


def test_project_drr_autograd():
    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    h_matrix_inv = torch.eye(4)
    h_matrix_inv.requires_grad = True
    source_distance = 1000.0
    output_size = torch.Size([10, 15])
    detector_spacing = torch.tensor([0.2, 0.25])
    res = autograd.project_drr(h_matrix_inv, input_, voxel_spacing, source_distance, output_size[1], output_size[0],
                               torch.zeros(2, dtype=torch.float64), detector_spacing)
    assert res.size() == output_size

    loss_grad = torch.zeros_like(res)
    loss_grad[5, 7] = 1.0
    res.backward(loss_grad)
    plt.imshow(res.detach().cpu().numpy())
    plt.show()
    print(h_matrix_inv.grad)
    epsilon = 1.0e-4
    print("epsilon =", epsilon)
    out = torch.empty_like(h_matrix_inv.detach())
    for j in range(4):
        for i in range(4):
            delta = torch.zeros((4, 4))
            delta[j, i] = epsilon
            h_matrix_inv2 = torch.eye(4) + delta
            res2 = autograd.project_drr(h_matrix_inv2, input_, voxel_spacing, source_distance, output_size[1],
                                        output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing)
            out[j, i] = (res2[5, 7] - res[5, 7]) / epsilon
    print(out)
