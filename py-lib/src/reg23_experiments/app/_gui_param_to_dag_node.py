from typing import Any
import traitlets
import logging

from reg23_experiments.data.structs import Cropping, Transformation
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph, dadg_updater

__all__ = ["mask_follows_transformation", "respond_to_mask_change", "respond_to_crop_change",
           "respond_to_crop_value_change", "respond_to_crop_value_value_change"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["mask_transformation"])
def mask_follows_transformation(*, current_transformation: Transformation) -> dict[str, Any]:
    return {"mask_transformation": current_transformation}


def respond_to_mask_change(*, dadg: DirectedAcyclicDataGraph, change) -> None:
    if change.new == "None":
        dadg.remove_updater("mask_follows_transformation")
        dadg.set("mask_transformation", None, check_equality=True)
    else:
        dadg.add_updater("mask_follows_transformation", mask_follows_transformation)


def respond_to_crop_change(*, dadg: DirectedAcyclicDataGraph, change, namespace: str | None = None) -> None:
    key = "cropping" if namespace is None else f"{namespace}__cropping"
    if change.new == "None":
        dadg.set(key, None)
    elif change.new == "fixed":
        dadg.set(key, change.owner.cropping_value)


def respond_to_crop_value_change(*, dadg: DirectedAcyclicDataGraph, change, namespace: str | None = None) -> None:
    if change.owner.cropping != "fixed":
        return
    assert isinstance(change.new, Cropping)
    key = "cropping" if namespace is None else f"{namespace}__cropping"
    dadg.set(key, change.new)
    change.new.observe(
        lambda _change, _dadg=dadg, _namespace=namespace: respond_to_crop_value_value_change(dadg=_dadg, change=_change,
                                                                                             namespace=_namespace))


def respond_to_crop_value_value_change(*, dadg: DirectedAcyclicDataGraph, change, namespace: str | None = None) -> None:
    key = "cropping" if namespace is None else f"{namespace}__cropping"
    dadg.set(key, change.owner)
