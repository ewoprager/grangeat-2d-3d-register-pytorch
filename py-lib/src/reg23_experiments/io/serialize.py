import logging
import traitlets
from typing import Any

import pathlib

logger = logging.getLogger(__name__)

__all__ = ["JsonSerializable", "serialize_recursive"]

type JsonSerializable = None | bool | int | float | str | list[JsonSerializable] | dict[str, JsonSerializable]


def serialize_recursive(value: Any) -> JsonSerializable:
    if isinstance(value, traitlets.HasTraits):
        return {k: serialize_recursive(v) for k, v in value.trait_values().items()}
    if isinstance(value, dict):
        return {k: serialize_recursive(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_recursive(e) for e in value]
    if isinstance(value, pathlib.Path) or value is None:
        return str(value)
    return value
