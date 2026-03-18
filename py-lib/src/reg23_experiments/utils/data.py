import traitlets
import copy
from typing import Any

__all__ = ["clone_has_traits"]


def clone_has_traits(obj: Any) -> Any:
    if isinstance(obj, traitlets.HasTraits):
        cls = obj.__class__
        kwargs = {}
        for name in obj.traits().keys():
            if name.startswith("_"):
                continue
            if obj.trait_has_value(name):
                kwargs[name] = clone_has_traits(getattr(obj, name))
        return cls(**kwargs)
    else:
        return copy.copy(obj)
