from typing import Any, Literal
import logging

from reg23_experiments.data.structs import Cropping, Transformation
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph, dadg_updater
from reg23_experiments.experiments.parameters import XrayParameters

__all__ = ["mask_follows_transformation", "respond_to_mask_change", "cropping_changed", "cropping_value_changed",
           "cropping_value_value_changed"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["mask_transformation"])
def mask_follows_transformation(*, current_transformation: Transformation) -> dict[str, Any]:
    return {"mask_transformation": current_transformation}


def respond_to_mask_change(*, dadg: DirectedAcyclicDataGraph, new_value: str) -> None:
    if new_value == "None":
        dadg.remove_updater("mask_follows_transformation")
        dadg.set("mask_transformation", None, check_equality=True)
    else:
        dadg.add_updater("mask_follows_transformation", mask_follows_transformation)


def cropping_changed(*, dadg: DirectedAcyclicDataGraph, new_value: Literal["None", "Fixed"], owner: XrayParameters,
                     namespace: str | None = None) -> None:
    key = "cropping" if namespace is None else f"{namespace}__cropping"
    if new_value == "None":
        dadg.set(key, None)
    elif new_value == "Fixed":
        dadg.set(key, owner.cropping_value)


def cropping_value_changed(*, dadg: DirectedAcyclicDataGraph, new_value: Cropping | None, owner: XrayParameters,
                           namespace: str | None = None) -> None:
    if owner.cropping != "Fixed":
        return
    assert isinstance(new_value, Cropping)
    key = "cropping" if namespace is None else f"{namespace}__cropping"
    dadg.set(key, new_value)
    new_value.observe(
        lambda _change, _dadg=dadg, _namespace=namespace: cropping_value_value_changed(dadg=_dadg, owner=_change.owner,
                                                                                       namespace=_namespace))


def cropping_value_value_changed(*, dadg: DirectedAcyclicDataGraph, owner: Cropping,
                                 namespace: str | None = None) -> None:
    key = "cropping" if namespace is None else f"{namespace}__cropping"
    dadg.set(key, owner)
