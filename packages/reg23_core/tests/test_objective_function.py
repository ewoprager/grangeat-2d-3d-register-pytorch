import torch

from reg23_core import objective_function, project_drr_cuboid_masks_batched, project_drrs_batched


def weighted_ncc(xs: torch.Tensor, ys: torch.Tensor, weights: torch.Tensor, *,
                 dim: int | torch.Size | tuple | None = None) -> torch.Tensor:
    size = xs.size()
    ret_size = size[:-2]
    dtype = xs.dtype
    device = xs.device
    assert ys.dtype == dtype
    assert ys.device == device
    assert weights.dtype == dtype
    assert weights.device == device
    if dim is None:
        dim = torch.Size(range(len(size)))  #
    # filtering out ZNCC calculations where fewer than 2 value pairs are being used
    exclude_mask = (weights.count_nonzero(dim=dim) < 2).expand(ret_size)
    if (exclude_mask.numel() - exclude_mask.count_nonzero()) < 1:
        return torch.zeros(ret_size, dtype=dtype, device=device)
    sum_w = weights.sum(dim=dim)
    sum_wx = (weights * xs).sum(dim=dim)
    sum_wy = (weights * ys).sum(dim=dim)
    sum_wx2 = (weights * xs.square()).sum(dim=dim)
    sum_wy2 = (weights * ys.square()).sum(dim=dim)
    sum_prod = (weights * xs * ys).sum(dim=dim)
    num = sum_w * sum_prod - sum_wx * sum_wy
    # make sure excluded values are non-negative for sqrt
    sum_wx[exclude_mask] = 0.0
    sum_wy[exclude_mask] = 0.0
    den = (sum_w * sum_wx2 - sum_wx.square()).sqrt() * (sum_w * sum_wy2 - sum_wy.square()).sqrt()
    ret = num / (den + 1e-10)
    # make sure excluded values are zero
    ret[exclude_mask] = 0.0
    return ret


def test_objective_function():
    if not torch.cuda.is_available():
        return
    input_ = torch.rand((11, 12, 8))
    voxel_spacing = torch.tensor([0.1, 0.2, 0.3])
    inv_hs = torch.stack([torch.eye(4)])
    source_distance = 1000.0
    output_size = torch.Size([3, 10, 15])
    fixed_image = torch.rand(output_size[-2:])
    detector_spacing = torch.tensor([0.2, 0.25])
    weight_alpha = 0.0
    res_of = objective_function(input_.cuda(), fixed_image.cuda(), voxel_spacing.cuda(), inv_hs.cuda(), source_distance,
                                torch.zeros(2, dtype=torch.float64), detector_spacing.cuda(), weight_alpha).cpu()

    drrs = project_drrs_batched(input_.cuda(), voxel_spacing.cuda(), inv_hs.cuda(), source_distance, output_size[-1],
                                output_size[-2], torch.zeros(2, dtype=torch.float64), detector_spacing.cuda())
    masks = project_drr_cuboid_masks_batched(torch.tensor(input_.size()).cuda(), voxel_spacing.cuda(), inv_hs.cuda(),
                                             source_distance, output_size[-1], output_size[-2],
                                             torch.zeros(2, dtype=torch.float64).cuda(), detector_spacing.cuda())
    if weight_alpha > 1e-10:
        weights = torch.pow(3.0 * masks * masks - 2.0 * masks * masks * masks, 1.0 / (weight_alpha * weight_alpha))
    else:
        weights = masks.clone()
        weights[weights < 1.0 - 1e-3] = 0.0
    masked_fixeds = masks * fixed_image.cuda().unsqueeze(0)
    res_og = weighted_ncc(drrs, masked_fixeds, weights, dim=(-1, -2)).cpu()

    print("New: ", res_of)
    print("Old: ", res_og)
