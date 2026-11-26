import os
from typing import Any

os.environ["QT_API"] = "PyQt6"

import torch
from jaxtyping import Float32

from reg23_experiments.ops.data_manager import dadg_updater

__all__ = ["update_ncc_sim_met_image", "update_gc_sim_met_image"]


@dadg_updater(names_returned=["ncc_sim_met_image"])
def update_ncc_sim_met_image(  #
    *,  #
    moving_image: Float32[torch.Tensor, "n m"],  #
    fixed_image: Float32[torch.Tensor, "n m"],  #
) -> dict[str, Any]:
    std, mean = torch.std_mean(moving_image)
    xs = (moving_image - mean) / std
    std, mean = torch.std_mean(fixed_image)
    ys = (fixed_image - mean) / std

    return {  #
        "ncc_sim_met_image": xs * ys,  #
    }


@dadg_updater(names_returned=["gc_sim_met_image"])
def update_gc_sim_met_image(  #
    *,  #
    moving_image: Float32[torch.Tensor, "n m"],  #
    fixed_image: Float32[torch.Tensor, "n m"],  #
) -> dict[str, Any]:
    device = moving_image.device
    dtype = moving_image.dtype
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=dtype, device=device)
    sobel_y = sobel_x.t()
    sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
    sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)

    xs = moving_image.unsqueeze(0).unsqueeze(0)
    ys = fixed_image.unsqueeze(0).unsqueeze(0)

    gx_xs = torch.nn.functional.conv2d(xs, sobel_x).squeeze()
    gy_xs = torch.nn.functional.conv2d(xs, sobel_y).squeeze()
    gx_ys = torch.nn.functional.conv2d(ys, sobel_x).squeeze()
    gy_ys = torch.nn.functional.conv2d(ys, sobel_y).squeeze()

    std, mean = torch.std_mean(gx_xs)
    gx_xs = (gx_xs - mean) / std
    std, mean = torch.std_mean(gy_xs)
    gy_xs = (gy_xs - mean) / std
    std, mean = torch.std_mean(gx_ys)
    gx_ys = (gx_ys - mean) / std
    std, mean = torch.std_mean(gy_ys)
    gy_ys = (gy_ys - mean) / std

    return {  #
        "gc_sim_met_image": 0.5 * (gx_xs * gx_ys + gy_xs * gy_ys),  #
    }
