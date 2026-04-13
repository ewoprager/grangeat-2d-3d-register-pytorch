from typing import Callable, NamedTuple

import traitlets

from reg23_experiments.utils.reflection import FunctionArgument, takes_positional_args

__all__ = ["Dependency", "NoNodeDataType", "NoNodeData", "Updater"]


class Dependency(traitlets.HasTraits):
    depender: str = traitlets.Unicode(allow_none=False)
    depended: str = traitlets.Unicode(allow_none=False)


class NoNodeDataType:
    pass


NoNodeData = NoNodeDataType()


class Updater(NamedTuple):
    function: Callable
    returned: list[str]
    arguments: list[FunctionArgument]
    dependencies: list[Dependency]

    @staticmethod
    def build(*, function: Callable, returned: list[str]) -> 'Updater':
        if takes_positional_args(function):
            raise ValueError(f"An Updater cannot be constructed from a function that takes positional arguments. "
                             f"Prepend arguments with `*` to ensure otherwise.")
        arguments = FunctionArgument.get_for_function(function)
        dependencies: list[Dependency] = []
        for depended in arguments:
            for depender in returned:
                dependencies.append(Dependency(depender=depender, depended=depended.name))
        return Updater(function=function, returned=returned, arguments=arguments, dependencies=dependencies)
