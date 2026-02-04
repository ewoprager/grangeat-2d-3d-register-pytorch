from typing import Any

import torch

from reg23_experiments.ops.geometry import get_crop_nonzero_drr, get_crop_full_depth_drr
from reg23_experiments.data.structs import Cropping, Transformation
from reg23_experiments.ops.data_manager import DAG, dag_updater

__all__ = ["mask_follows_transformation", "cropping_follows_nonzero_drr", "cropping_follows_full_depth_drr",
           "respond_to_mask_change", "respond_to_crop_change"]


@dag_updater(names_returned=["mask_transformation"])
def mask_follows_transformation(current_transformation: Transformation) -> dict[str, Any]:
    return {"mask_transformation": current_transformation}


@dag_updater(names_returned=["cropping"])
def cropping_follows_nonzero_drr(image_2d_full: torch.Tensor, source_distance: float,
                                 current_transformation: Transformation,
                                 ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor,
                                 fixed_image_spacing: torch.Tensor) -> dict[str, Any]:
    return {"cropping": get_crop_nonzero_drr(image_2d_full=image_2d_full, source_distance=source_distance,
                                             current_transformation=current_transformation, ct_volumes=ct_volumes,
                                             ct_spacing=ct_spacing, fixed_image_spacing=fixed_image_spacing)}


@dag_updater(names_returned=["cropping"])
def cropping_follows_full_depth_drr(image_2d_full: torch.Tensor, source_distance: float,
                                    current_transformation: Transformation,
                                    ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor,
                                    fixed_image_spacing: torch.Tensor) -> dict[str, Any]:
    return {"cropping": get_crop_full_depth_drr(image_2d_full=image_2d_full, source_distance=source_distance,
                                                current_transformation=current_transformation, ct_volumes=ct_volumes,
                                                ct_spacing=ct_spacing, fixed_image_spacing=fixed_image_spacing)}


def respond_to_mask_change(dag: DAG, change) -> None:
    if change.new == "None":
        dag.remove_updater("mask_follows_transformation")
        dag.set_data("mask_transformation", None, check_equality=True)
    else:
        dag.add_updater("mask_follows_transformation", mask_follows_transformation)


def respond_to_crop_change(dag: DAG, change) -> None:
    if change.new == "None":
        dag.remove_updater("cropping_follows_transformation")
        dag.set_data("cropping", None)
    elif change.new == "nonzero_drr":
        dag.remove_updater("cropping_follows_transformation")
        dag.add_updater("cropping_follows_transformation", cropping_follows_nonzero_drr)
    elif change.new == "full_depth_drr":
        dag.remove_updater("cropping_follows_transformation")
        dag.add_updater("cropping_follows_transformation", cropping_follows_full_depth_drr)
