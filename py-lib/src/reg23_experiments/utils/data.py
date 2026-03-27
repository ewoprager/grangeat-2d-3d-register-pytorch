import copy
from typing import Any, Callable

import traitlets

__all__ = ["clone_has_traits", "observe_all_traits_recursively"]


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


def observe_all_traits_recursively(callback: Callable, obj: traitlets.HasTraits) -> None:
    obj.observe(callback)
    for name, trait in obj.traits().items():
        value = getattr(obj, name)
        if isinstance(value, traitlets.HasTraits):
            observe_all_traits_recursively(callback, value)
        elif isinstance(value, list):
            for e in value:
                if isinstance(e, traitlets.HasTraits):
                    observe_all_traits_recursively(callback, e)
        elif isinstance(value, dict):
            for v in value.values():
                if isinstance(v, traitlets.HasTraits):
                    observe_all_traits_recursively(callback, v)
