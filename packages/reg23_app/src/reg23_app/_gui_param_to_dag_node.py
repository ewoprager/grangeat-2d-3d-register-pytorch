import logging
from typing import Any, Literal

from reg23_experiments.data.structs import Cropping, Error, Transformation
from reg23_experiments.experiments.parameters import XrayParameters
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph, capture_in_namespaces, dadg_updater

__all__ = ["mask_follows_transformation", "respond_to_mask_change", "cropping_changed", "cropping_value_changed",
           "cropping_value_value_changed"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["mask_transformation"])
def mask_follows_transformation(*, current_transformation: Transformation) -> dict[str, Any]:
    return {"mask_transformation": current_transformation}


def respond_to_mask_change(*, dadg: DirectedAcyclicDataGraph, new_value: str, namespace_captures: dict[str, str],
                           namespace: str) -> None:
    if new_value == "None":
        logger.debug("Removing mask_transformation updates")
        if isinstance(err := dadg.remove_updater(f"{namespace}__mask_follows_transformation"), Error):
            logger.error(f"Failed to remove mask_transformation updater: {err.description}")
        if isinstance(err := dadg.set(f"{namespace}__mask_transformation", None, check_equality=True), Error):
            logger.error(f"Failed to set mask_transformation: {err.description}")
    else:
        logger.debug("Adding mask_transformation updates")
        if isinstance(err := dadg.add_updater(  #
                f"{namespace}__mask_follows_transformation",  #
                capture_in_namespaces(namespace_captures)(mask_follows_transformation)  #
        ), Error):
            logger.error(f"Failed to add mask_transformation updater: {err.description}")


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
