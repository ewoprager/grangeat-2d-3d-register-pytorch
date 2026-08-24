import logging
from typing import Literal

from reg23_experiments.data.parameters import XrayParameters
from reg23_experiments.data.structs import Cropping
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph

__all__ = ["cropping_changed", "cropping_value_changed", "cropping_value_value_changed"]

logger = logging.getLogger(__name__)


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
