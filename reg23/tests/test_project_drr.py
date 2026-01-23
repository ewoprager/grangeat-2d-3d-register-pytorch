import pytest
import torch

from reg23 import *


def test_project_drr():
    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    h_matrix_inv = torch.eye(4)
    source_distance = 1000.0
    output_size = torch.Size([10, 15])
    detector_spacing = torch.tensor([0.2, 0.25])
    res = project_drr(input_, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                      torch.zeros(2, dtype=torch.float64), detector_spacing)
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(res.cpu().numpy())
    # axes[0].set_title("cpu")
    assert res.size() == output_size
    if torch.cuda.is_available():
        res_cuda = project_drr(input_.cuda(), voxel_spacing.cuda(), h_matrix_inv.cuda(), source_distance,
                               output_size[1], output_size[0], torch.zeros(2, dtype=torch.float64),
                               detector_spacing.cuda())
        # axes[1].imshow(res_cuda.cpu().numpy())
        # axes[1].set_title("cuda")
        assert res == pytest.approx(res_cuda.cpu(), abs=0.01)
    # plt.show()

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
    display = True

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    source_distance = 1000.0
    output_size = torch.Size([10, 15])
    detector_spacing = torch.tensor([0.2, 0.25])
    if display:
        fig, axes = plt.subplots(len(devices), 1)
        if len(devices) < 2:
            axes = [axes]

    for device_index, device_name in enumerate(devices):
        device = torch.device(device_name)
        volume = input_.to(device=device)
        h_matrix_inv = torch.eye(4, device=device)
        h_matrix_inv.requires_grad = True
        res = project_drr(volume, voxel_spacing, h_matrix_inv, source_distance, output_size[1], output_size[0],
                          torch.zeros(2, dtype=torch.float64), detector_spacing)
        assert res.size() == output_size

        # loss_grad = torch.zeros_like(res)
        # loss_grad[5, 7] = 1.0
        res.backward(torch.ones_like(res))
        epsilon = 1.0e-4
        if display:
            print(device_name)
            print(h_matrix_inv.grad)
            axes[device_index].imshow(res.detach().cpu().numpy())
            axes[device_index].set_title(device_name)
            print("epsilon =", epsilon)
        out = torch.empty_like(h_matrix_inv)
        out.requires_grad = False
        for j in range(4):
            for i in range(4):
                h_matrix_inv_delta = h_matrix_inv.clone().detach()
                h_matrix_inv_delta[j, i] += epsilon
                res2 = project_drr(volume, voxel_spacing, h_matrix_inv_delta, source_distance, output_size[1],
                                   output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing)
                out[i, j] = (res2.sum() - res.sum()) / epsilon
        if display:
            print(out)  # assert h_matrix_inv.grad == pytest.approx(out.detach().cpu(), abs=0.001, rel=0.01)
    if display:
        plt.show()


def test_project_drrs_batched():
    if not torch.cuda.is_available():
        return
    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    inv_hs = torch.stack([torch.eye(4), torch.eye(4), torch.eye(4)])
    source_distance = 1000.0
    output_size = torch.Size([3, 10, 15])
    detector_spacing = torch.tensor([0.2, 0.25])
    res_cuda = project_drrs_batched(input_.cuda(), voxel_spacing.cuda(), inv_hs.cuda(), source_distance, output_size[1],
                                    output_size[0], torch.zeros(2, dtype=torch.float64), detector_spacing.cuda())
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(res_cuda[0].cpu().numpy())
    axes[1].imshow(res_cuda[1].cpu().numpy())
    axes[2].imshow(res_cuda[2].cpu().numpy())
    assert res_cuda.size() == output_size
    plt.show()
