from typing import NamedTuple, Callable

from reg23_experiments.utils.reflection import FunctionArgument

__all__ = ["Dependency", "NoNodeDataType", "NoNodeData", "Updater"]


class Dependency(NamedTuple):
    depender: str
    depended: str


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
        arguments = FunctionArgument.get_for_function(function)
        dependencies: list[Dependency] = []
        for depended in arguments:
            for depender in returned:
                dependencies.append(Dependency(depender=depender, depended=depended.name))
        return Updater(function=function, returned=returned, arguments=arguments, dependencies=dependencies)
