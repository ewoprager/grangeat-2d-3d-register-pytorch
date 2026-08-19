"""
Similarity metrics to use for image registration.

All similarity metrics take exactly two positional arguments: xs and ys. They must be tensors that are broadcastable
together.

All similarity metrics optionally take the keyword argument 'weights' which is a tensor or None by default. If given,
xs, ys and weights must be broadcastable together.
"""

import logging
from typing import Literal

import torch

__all__ = ["ncc", "gradient_correlation", "mutual_information"]

logger = logging.getLogger(__name__)


def _broadcast_similarity_tensors(  #
        xs: torch.Tensor,  #
        ys: torch.Tensor,  #
        weights: torch.Tensor | None  #
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Size, torch.dtype, torch.device]:
    # check tensor compatibility
    dtype = xs.dtype
    device = xs.device
    assert ys.dtype == dtype
    assert ys.device == device
    if weights is not None:
        assert weights.dtype == dtype
        assert weights.device == device
    # broadcast all tensors to the same shape
    size = torch.broadcast_shapes(  #
        xs.size(), ys.size()  #
    ) if weights is None else torch.broadcast_shapes(  #
        xs.size(), ys.size(), weights.size()  #
    )
    xs = xs.broadcast_to(size)
    ys = ys.broadcast_to(size)
    if weights is not None:
        weights = weights.broadcast_to(size)
    return xs, ys, weights, size, dtype, device


def ncc(  #
        xs: torch.Tensor,  #
        ys: torch.Tensor,  #
        *,  #
        weights: torch.Tensor | None = None,  #
        dim: int | torch.Size | tuple | None = None,  #
) -> torch.Tensor:
    xs, ys, weights, size, dtype, device = _broadcast_similarity_tensors(xs, ys, weights)
    # convert the given `dim` parameter to a torch.Size with non-negative elements
    if dim is None:
        dim = torch.Size(range(len(size)))  #
    elif isinstance(dim, int):
        dim = torch.Size([dim % len(size)])
    else:
        dim = torch.Size(d % len(size) for d in dim)
    # determine the size of the returned value
    ret_size = torch.Size([s for i, s in enumerate(size) if i not in dim])
    if weights is None:
        sum_x = xs.sum(dim=dim)
        sum_y = ys.sum(dim=dim)
        sum_x2 = xs.square().sum(dim=dim)
        sum_y2 = ys.square().sum(dim=dim)
        sum_prod = (xs * ys).sum(dim=dim)
        n = float(xs.numel() // sum_x.numel())
        num = n * sum_prod - sum_x * sum_y
        den = (n * sum_x2 - sum_x.square()).sqrt() * (n * sum_y2 - sum_y.square()).sqrt()
        ret = num / (den + 1e-10)
    else:
        # filtering out ZNCC calculations where fewer than 2 value pairs are being used
        exclude_mask = weights.count_nonzero(dim=dim) < 2
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


def gradient_correlation(  #
        xs: torch.Tensor,  #
        ys: torch.Tensor,  #
        *,  #
        weights: torch.Tensor | None = None,  #
        gradient_method: Literal["sobel", "central_difference"] = "sobel",  #
) -> torch.Tensor:
    """
    Evaluate the gradient correlation between the given tensors over the last two dimensions.
    :param xs:
    :param ys:
    :param weights:
    :param gradient_method:
    :return:
    """
    xs, ys, weights, size, dtype, device = _broadcast_similarity_tensors(xs, ys, weights)

    assert len(size) >= 2
    ret_size = size[:-2]
    # make sure there is a batch dimension so we can safely flatten them
    if len(size) < 3:
        size = torch.Size([1, *size])
    # broadcast all tensors to the same shape
    xs = xs.broadcast_to(size)
    ys = ys.broadcast_to(size)
    if weights is not None:
        weights = weights.broadcast_to(size)
    # flatten batch dimensions
    xs = xs.flatten(end_dim=-3)
    ys = ys.flatten(end_dim=-3)
    if weights is not None:
        weights = weights.flatten(end_dim=-3)
    weights_x = weights
    weights_y = weights
    # size is now (N total batches, H, W)
    if gradient_method == "sobel":
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=dtype, device=device)
        sobel_y = sobel_x.t()
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
        xs = xs.unsqueeze(1)
        ys = ys.unsqueeze(1)
        gx_xs = torch.nn.functional.conv2d(xs, sobel_x)[0]
        gy_xs = torch.nn.functional.conv2d(xs, sobel_y)[0]
        gx_ys = torch.nn.functional.conv2d(ys, sobel_x)[0]
        gy_ys = torch.nn.functional.conv2d(ys, sobel_y)[0]
        if weights is not None:
            # take the geometric means of the weights that contribute to each value in the gradient images
            log_weights = weights.unsqueeze(1).log()
            kernel_sum = sobel_x.abs().sum()
            weights_x = (torch.nn.functional.conv2d(log_weights, sobel_x.abs())[0] / kernel_sum).exp()
            weights_y = (torch.nn.functional.conv2d(log_weights, sobel_y.abs())[0] / kernel_sum).exp()
    else:  # gradient_method is "central_difference"
        gx_xs = torch.gradient(xs, dim=-2)[0]
        gy_xs = torch.gradient(xs, dim=-1)[0]
        gx_ys = torch.gradient(ys, dim=-2)[0]
        gy_ys = torch.gradient(ys, dim=-1)[0]
    return 0.5 * (  #
            ncc(gx_xs, gx_ys, weights=weights_x, dim=(-2, -1)) +  #
            ncc(gy_xs, gy_ys, weights=weights_y, dim=(-2, -1))  #
    ).view(ret_size)


def mutual_information(  #
        xs: torch.Tensor,  #
        ys: torch.Tensor,  #
        *,  #
        weights: torch.Tensor | None = None,  #
        dim: int | tuple | torch.Size | None = None,  #
        x_bins: int = 64,  #
        y_bins: int = 64,  #
) -> torch.Tensor:
    """
    Evaluate the mutual information between xs and ys along the given dimensions.

    A weighting tensor can optionally be given to afford more importance to some values than others.

    :param xs:
    :param ys:
    :param weights: (optional) a tensor with which to weight the evaluation
    :param dim: (default: all) dimensions along which to calculate the MI; all dimensions are used if not given
    :param x_bins: (default: 64) The number of equally-spaced bins in which to bin the values in xs
    :param y_bins: (default: 64) The number of equally-spaced bins in which to bin the values in ys
    :return:
    """
    xs, ys, weights, size, dtype, device = _broadcast_similarity_tensors(xs, ys, weights)
    if dim is None:
        dim = torch.Size(range(len(size)))  #
    elif isinstance(dim, int):
        dim = torch.Size([dim % len(size)])
    else:
        dim = torch.Size(d % len(size) for d in dim)
    # determine the size of the returned value
    ret_size = torch.Size([s for i, s in enumerate(size) if i not in dim])
    ret_dims = torch.Size([i for i in range(len(size)) if i not in dim])
    # filtering out ZNCC calculations where fewer than 2 value pairs are being used
    if weights is not None:
        exclude_mask = weights.count_nonzero(dim=dim) < 2
        if (exclude_mask.numel() - exclude_mask.count_nonzero()) < 1:
            return torch.zeros(ret_size, dtype=dtype, device=device)

    x_min = xs.amin(dim=dim, keepdim=True)
    x_max = xs.amax(dim=dim, keepdim=True)
    y_min = ys.amin(dim=dim, keepdim=True)
    y_max = ys.amax(dim=dim, keepdim=True)

    x_bins_f = (xs - x_min) / (x_max - x_min) * (x_bins - 1)
    y_bins_f = (ys - y_min) / (y_max - y_min) * (y_bins - 1)

    # moving the chosen dimensions to the back and flattening them
    x_bins_f = x_bins_f.permute(*ret_dims, *dim).flatten(start_dim=-len(dim))
    y_bins_f = y_bins_f.permute(*ret_dims, *dim).flatten(start_dim=-len(dim))
    if weights is not None:
        weights = weights.permute(*ret_dims, *dim).flatten(start_dim=-len(dim))

    x_bins_0 = x_bins_f.floor().long().clamp(max=x_bins - 2)
    y_bins_0 = y_bins_f.floor().long().clamp(max=y_bins - 2)

    x_bins_1 = x_bins_0 + 1
    y_bins_1 = y_bins_0 + 1

    x_alphas = x_bins_f - x_bins_0.float()
    y_alphas = y_bins_f - y_bins_0.float()

    contributions_00 = (1.0 - x_alphas) * (1.0 - y_alphas)
    contributions_01 = (1.0 - x_alphas) * y_alphas
    contributions_10 = x_alphas * (1.0 - y_alphas)
    contributions_11 = x_alphas * y_alphas

    if weights is not None:
        contributions_00 *= weights
        contributions_01 *= weights
        contributions_10 *= weights
        contributions_11 *= weights

    bins_00 = x_bins_0 * y_bins + y_bins_0
    bins_01 = x_bins_0 * y_bins + y_bins_1
    bins_10 = x_bins_1 * y_bins + y_bins_0
    bins_11 = x_bins_1 * y_bins + y_bins_1

    hist = torch.zeros((*ret_size, x_bins * y_bins), dtype=dtype, device=device)
    hist.scatter_add_(-1, bins_00, contributions_00)
    hist.scatter_add_(-1, bins_01, contributions_01)
    hist.scatter_add_(-1, bins_10, contributions_10)
    hist.scatter_add_(-1, bins_11, contributions_11)
    hist = hist.reshape(*ret_size, x_bins, y_bins)

    p = hist / hist.sum(dim=(-2, -1), keepdim=True)
    px = p.sum(dim=-2, keepdim=True)
    py = p.sum(dim=-1, keepdim=True)
    return (p * (p / (px * py)).log()).nan_to_num().sum(dim=(-2, -1))


if False:
    def local_ncc(xs: torch.Tensor, ys: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
        """
        Divides the two input images into patches of size `kernel_size`, evaluates the ZNCC between each
        corresponding pair
        of patches and returns the mean of the resulting NCC values over all the patches.

        :param xs: [tensor of size (n, m)] one input image
        :param ys: [tensor of size (n, m)] another input image
        :param kernel_size:
        :return: The mean of the ZNCCs of the pairs of corresponding image patches.
        """
        # check tensor compatibility
        dtype = xs.dtype
        device = xs.device
        assert ys.dtype == dtype
        assert ys.device == device
        # broadcast all tensors to the same shape
        size = torch.broadcast_shapes(xs.size(), ys.size())
        xs = xs.broadcast_to(size)
        ys = ys.broadcast_to(size)
        assert len(size) == 2
        xs_patches = torch.nn.functional.unfold(xs.unsqueeze(0), kernel_size=kernel_size,
                                                stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
        ys_patches = torch.nn.functional.unfold(ys.unsqueeze(0), kernel_size=kernel_size,
                                                stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
        return ncc(xs_patches, ys_patches, dim=0).mean()


    def multiscale_ncc(xs: torch.Tensor, ys: torch.Tensor, *, kernel_size: int, llambda: float) -> torch.Tensor:
        return ncc(xs, ys) + llambda * local_ncc(xs, ys, kernel_size=kernel_size)


    def weighted_local_ncc(xs: torch.Tensor, ys: torch.Tensor, weights: torch.Tensor, *,
                           kernel_size: int) -> torch.Tensor:
        """
        Divides the two input images into patches of size `kernel_size`, evaluates the WZNCC between each
        corresponding pair
        of patches and returns the mean of the resulting NCC values over all the patches.

        :param xs: [tensor of size (n, m)] one input image
        :param ys: [tensor of size (n, m)] another input image
        :param weights: [tensor of size (n, m)] the image of weights
        :param kernel_size:
        :return: The mean of the WZNCCs of the pairs of corresponding image patches.
        """
        assert xs.size() == ys.size()
        assert len(xs.size()) == 2
        xs_patches = torch.nn.functional.unfold(xs.unsqueeze(0), kernel_size=kernel_size,
                                                stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
        ys_patches = torch.nn.functional.unfold(ys.unsqueeze(0), kernel_size=kernel_size,
                                                stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
        ws_patches = torch.nn.functional.unfold(weights.unsqueeze(0), kernel_size=kernel_size,
                                                stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
        patch_wznccs = weighted_ncc(xs_patches, ys_patches, ws_patches, dim=0)  # size = (patch number)
        patch_weights = ws_patches.mean(dim=0)  # size = (patch number)
        # return (patch_weights * patch_wznccs).sum() / patch_weights.sum()
        ret = (patch_weights * patch_wznccs).sum() / patch_weights.sum()
        return ret
