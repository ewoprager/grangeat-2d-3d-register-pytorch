from typing import Any
import traitlets
import logging

import torch

from reg23_experiments.ops.geometry import get_crop_nonzero_drr, get_crop_full_depth_drr
from reg23_experiments.data.structs import Cropping, Transformation
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph, dadg_updater

__all__ = ["mask_follows_transformation", "cropping_follows_nonzero_drr", "cropping_follows_full_depth_drr",
           "respond_to_mask_change", "respond_to_crop_change", "respond_to_crop_value_change",
           "respond_to_crop_value_value_change"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["mask_transformation"])
def mask_follows_transformation(current_transformation: Transformation) -> dict[str, Any]:
    return {"mask_transformation": current_transformation}


@dadg_updater(names_returned=["cropping"])
def cropping_follows_nonzero_drr(image_2d_full: torch.Tensor, source_distance: float,
                                 current_transformation: Transformation, ct_volumes: list[torch.Tensor],
                                 ct_spacing: torch.Tensor, fixed_image_spacing: torch.Tensor) -> dict[str, Any]:
    return {"cropping": get_crop_nonzero_drr(image_2d_full=image_2d_full, source_distance=source_distance,
                                             current_transformation=current_transformation, ct_volumes=ct_volumes,
                                             ct_spacing=ct_spacing, fixed_image_spacing=fixed_image_spacing)}


@dadg_updater(names_returned=["cropping"])
def cropping_follows_full_depth_drr(image_2d_full: torch.Tensor, source_distance: float,
                                    current_transformation: Transformation, ct_volumes: list[torch.Tensor],
                                    ct_spacing: torch.Tensor, fixed_image_spacing: torch.Tensor) -> dict[str, Any]:
    return {"cropping": get_crop_full_depth_drr(image_2d_full=image_2d_full, source_distance=source_distance,
                                                current_transformation=current_transformation, ct_volumes=ct_volumes,
                                                ct_spacing=ct_spacing, fixed_image_spacing=fixed_image_spacing)}


def respond_to_mask_change(dadg: DirectedAcyclicDataGraph, change) -> None:
    if change.new == "None":
        dadg.remove_updater("mask_follows_transformation")
        dadg.set("mask_transformation", None, check_equality=True)
    else:
        dadg.add_updater("mask_follows_transformation", mask_follows_transformation)


def respond_to_crop_change(dadg: DirectedAcyclicDataGraph, change) -> None:
    if change.new == "None":
        dadg.remove_updater("cropping_follows_transformation")
        dadg.set("cropping", None)
    elif change.new == "nonzero_drr":
        dadg.remove_updater("cropping_follows_transformation")
        dadg.add_updater("cropping_follows_transformation", cropping_follows_nonzero_drr)
    elif change.new == "full_depth_drr":
        dadg.remove_updater("cropping_follows_transformation")
        dadg.add_updater("cropping_follows_transformation", cropping_follows_full_depth_drr)
    elif change.new == "fixed":
        dadg.remove_updater("cropping_follows_transformation")
        dadg.set("cropping", change.owner.cropping_value)


def respond_to_crop_value_change(dadg: DirectedAcyclicDataGraph, change) -> None:
    if change.owner.cropping != "fixed":
        return
    assert isinstance(change.new, Cropping)
    dadg.set("cropping", change.new)
    change.new.observe(lambda _change: respond_to_crop_value_value_change(dadg, _change), names=traitlets.All)


def respond_to_crop_value_value_change(dadg: DirectedAcyclicDataGraph, change) -> None:
    dadg.set("cropping", change.owner)
