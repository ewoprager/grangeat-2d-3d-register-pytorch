import logging
from typing import Any

import numpy as np
import torch
from jaxtyping import Float64

from reg23_experiments.ops.data_manager import dadg_updater

__all__ = ["transform_layer_ct_fiducials", "ct_fiducial_world_to_layer", "transform_layer_fiducials"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["ct_fiducial_points"])
def transform_layer_ct_fiducials(*, layer_ct_fiducial_points: np.ndarray, ct_spacing: Float64[torch.Tensor, "3"],
                                 untruncated_ct_size: torch.Size) -> dict[str, Any]:
    """
    layer_ct_fiducial_points is set directly from the layer data, so is a numpy array of (z, y, x) at mm scale? with
    origin at top left.
    :param layer_ct_fiducial_points:
    :param ct_spacing:
    :param untruncated_ct_size:
    :return:
    """
    offset = 0.5 * ct_spacing * torch.tensor(untruncated_ct_size, dtype=torch.float64).flip(dims=(0,))
    return {"ct_fiducial_points": torch.tensor(layer_ct_fiducial_points).flip(dims=(1,)) - offset}


def ct_fiducial_world_to_layer(*, ct_fiducial_points: torch.Tensor, ct_spacing: Float64[torch.Tensor, "3"],
                               untruncated_ct_size: torch.Size) -> np.ndarray:
    offset = 0.5 * ct_spacing * torch.tensor(untruncated_ct_size, dtype=torch.float64).flip(dims=(0,))
    return (ct_fiducial_points + offset).flip(dims=(1,)).numpy()


@dadg_updater(names_returned=["fiducial_points"])
def transform_layer_fiducials(*, layer_fiducial_points: np.ndarray, fiducial_names: list[str],
                              ct_fiducial_names: list[str], image_2d_full_spacing: torch.Tensor,
                              image_2d_full_size: torch.Size) -> dict[str, Any]:
    offset = 0.5 * image_2d_full_spacing * torch.tensor(image_2d_full_size, dtype=torch.float64).flip(dims=(0,))
    return {"fiducial_points"}
