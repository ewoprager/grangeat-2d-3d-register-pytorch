from abc import ABC, abstractmethod
from typing import Any, Callable
import traitlets
import copy

from reg23_experiments.data.structs import Error
from reg23_experiments.utils.reflection import FunctionArgument

from ._data import NoNodeData, Updater

__all__ = ["Node", "DirectedAcyclicDataGraph", "ChildDirectedAcyclicDataGraph"]


class Node(traitlets.HasTraits):
    dependents: list[str] = traitlets.List(trait=traitlets.Unicode(), default_value=[])
    dirty: bool = traitlets.Bool(default_value=False)
    updater: str | None = traitlets.Unicode(allow_none=True, default_value=None)
    lazily_evaluated: bool = traitlets.Bool(default_value=True)
    data: Any = traitlets.Any(allow_none=True, default_value=NoNodeData)
    observers: dict[str, Callable] = traitlets.Dict(key_trait=traitlets.Unicode(), value_trait=traitlets.Callable())

    def __str__(self) -> str:
        ret = "Node(\n"
        ret += "\n\t".join(f"{name}={getattr(self, name)!r}" for name in self.trait_names() if not name.startswith('_'))
        ret += "\n)\n"
        return ret

    @traitlets.validate("dirty")
    def _valid_dirty(self, proposal):
        if self.updater is None and proposal["value"]:
            raise traitlets.TraitError("A node with no updater cannot be dirty.")
        return proposal["value"]

    def copy(self) -> 'Node':
        return Node(  #
            dependents=copy.deepcopy(self.dependents),  #
            dirty=self.dirty,  #
            updater=copy.deepcopy(self.updater),  #
            lazily_evaluated=self.lazily_evaluated,  #
            data=copy.deepcopy(self.data),  #
            observers=dict(),  #
        )

    def copy_with_new_data(self, data) -> 'Node':
        return Node(  #
            dependents=copy.deepcopy(self.dependents),  #
            dirty=self.dirty,  #
            updater=copy.deepcopy(self.updater),  #
            lazily_evaluated=self.lazily_evaluated,  #
            data=data,  #
            observers=dict(),  #
        )


class DirectedAcyclicDataGraph(ABC):
    @abstractmethod
    def has_node(self, node_name: str) -> bool:
        pass

    @abstractmethod
    def get(self, node_name: str, *, soft: bool = False) -> Any | Error:
        pass

    @abstractmethod
    def set(self, node_name: str, data: Any, *, check_equality: bool = False) -> None | Error:
        pass

    @abstractmethod
    def set_multiple(self, **kwargs) -> None | Error:
        pass

    @abstractmethod
    def set_evaluation_laziness(self, node_name: str, *, lazily_evaluated: bool) -> None:
        pass

    @abstractmethod
    def observe(self, node_name: str, observer_name: str, callback: Callable[[Any], None]) -> None:
        pass

    @abstractmethod
    def remove_observer(self, node_name: str, observer_name: str) -> None | Error:
        pass

    @abstractmethod
    def add_updater(self, updater_name: str, updater: Updater):
        pass

    @abstractmethod
    def remove_updater(self, updater_name: str) -> None | Error:
        pass

    @abstractmethod
    def get_with_args(self, args: list[FunctionArgument]) -> dict[str, Any] | Error:
        pass

    #### Protected properties and methods for access by ChildDADG

    @abstractmethod
    def _get_node_if_exists(self, node_name: str) -> Node | Error:
        pass

    @property
    @abstractmethod
    def _nodes(self) -> dict[str, Node]:
        pass

    @property
    @abstractmethod
    def _updaters(self) -> dict[str, Updater]:
        pass

    def _add_child(self, child: 'ChildDirectedAcyclicDataGraph'):
        pass


class ChildDirectedAcyclicDataGraph(DirectedAcyclicDataGraph):
    #### Protected properties and methods for access by parent DADG
    @abstractmethod
    def _make_copy(self, node_name: str) -> None | Error:
        pass
