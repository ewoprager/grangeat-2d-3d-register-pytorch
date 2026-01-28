from typing import Tuple, Literal
import logging

import torch

__all__ = ["ncc", "local_ncc", "multiscale_ncc", "weighted_ncc", "weighted_local_ncc", "gradient_correlation", ]

logger = logging.getLogger(__name__)


def ncc(xs: torch.Tensor, ys: torch.Tensor, *, dim: int | Tuple | torch.Size | None = None) -> torch.Tensor:
    assert xs.size() == ys.size()
    if dim is None:
        dim = torch.Size(range(len(xs.size())))
    sum_x = xs.sum(dim=dim)
    sum_y = ys.sum(dim=dim)
    sum_x2 = xs.square().sum(dim=dim)
    sum_y2 = ys.square().sum(dim=dim)
    sum_prod = (xs * ys).sum(dim=dim)
    n = float(xs.numel() // sum_x.numel())
    num = n * sum_prod - sum_x * sum_y
    den = (n * sum_x2 - sum_x.square()).sqrt() * (n * sum_y2 - sum_y.square()).sqrt()
    return num / (den + 1e-10)


def local_ncc(xs: torch.Tensor, ys: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    """
    Divides the two input images into patches of size `kernel_size`, evaluates the ZNCC between each corresponding pair
    of patches and returns the mean of the resulting NCC values over all the patches.

    :param xs: [tensor of size (n, m)] one input image
    :param ys: [tensor of size (n, m)] another input image
    :param kernel_size:
    :return: The mean of the ZNCCs of the pairs of corresponding image patches.
    """
    assert xs.size() == ys.size()
    assert len(xs.size()) == 2
    xs_patches = torch.nn.functional.unfold(xs.unsqueeze(0), kernel_size=kernel_size,
                                            stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
    ys_patches = torch.nn.functional.unfold(ys.unsqueeze(0), kernel_size=kernel_size,
                                            stride=kernel_size)  # size = (kernel_size * kernel_size, patch number)
    return ncc(xs_patches, ys_patches, dim=0).mean()


def multiscale_ncc(xs: torch.Tensor, ys: torch.Tensor, *, kernel_size: int, llambda: float) -> torch.Tensor:
    return ncc(xs, ys) + llambda * local_ncc(xs, ys, kernel_size=kernel_size)


def weighted_ncc(xs: torch.Tensor, ys: torch.Tensor, weights: torch.Tensor, *,
                 dim: int | torch.Size | Tuple | None = None) -> torch.Tensor:
    assert ys.size() == xs.size()
    assert weights.size() == xs.size()
    if dim is None:
        dim = torch.Size(range(len(xs.size())))
    sum_w = weights.sum(dim=dim)
    sum_wx = (weights * xs).sum(dim=dim)
    sum_wy = (weights * ys).sum(dim=dim)
    sum_wx2 = (weights * xs.square()).sum(dim=dim)
    sum_wy2 = (weights * ys.square()).sum(dim=dim)
    sum_prod = (weights * xs * ys).sum(dim=dim)
    num = sum_w * sum_prod - sum_wx * sum_wy
    den = (sum_w * sum_wx2 - sum_wx.square()).sqrt() * (sum_w * sum_wy2 - sum_wy.square()).sqrt()
    return num / (den + 1e-10)


def weighted_local_ncc(xs: torch.Tensor, ys: torch.Tensor, weights: torch.Tensor, *, kernel_size: int) -> torch.Tensor:
    """
    Divides the two input images into patches of size `kernel_size`, evaluates the WZNCC between each corresponding pair
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
    return (patch_weights * patch_wznccs).sum() / patch_weights.sum()


def gradient_correlation(xs: torch.Tensor, ys: torch.Tensor, *,
                         gradient_method: Literal["sobel", "central_difference"] = "sobel") -> torch.Tensor:
    assert xs.size() == ys.size()
    assert len(xs.size()) == 2
    if gradient_method == "sobel":
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        gx_xs = torch.nn.functional.conv2d(xs.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0))[0, 0]
        gy_xs = torch.nn.functional.conv2d(xs.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0))[0, 0]
        gx_ys = torch.nn.functional.conv2d(ys.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0))[0, 0]
        gy_ys = torch.nn.functional.conv2d(ys.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0))[0, 0]
    else:  # gradient_method is "central_difference"
        gx_xs = torch.gradient(xs, dim=0)[0]
        gy_xs = torch.gradient(xs, dim=1)[0]
        gx_ys = torch.gradient(ys, dim=0)[0]
        gy_ys = torch.gradient(ys, dim=1)[0]
    return 0.5 * (ncc(gx_xs, gx_ys) + ncc(gy_xs, gy_ys))
