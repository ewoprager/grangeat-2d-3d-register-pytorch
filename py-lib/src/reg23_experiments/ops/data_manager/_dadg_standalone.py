import collections
import functools
import logging
from typing import Any, Callable
import weakref
from traitlets.config import SingletonConfigurable
import traitlets

from reg23_experiments.data.structs import Error
from reg23_experiments.utils.reflection import FunctionArgument

from ._data import NoNodeData, Updater, Dependency
from ._i_directed_acyclic_data_graph import Node, IDirectedAcyclicDataGraph, IChildDirectedAcyclicDataGraph

logger = logging.getLogger(__name__)

__all__ = ["StandaloneDADG", "StandaloneDADGSingleton"]


class StandaloneDADG(IDirectedAcyclicDataGraph):
    """
    A data manager that stores data in a directed, acyclic graph.

    Each variable (piece of data) has a `str` name and is stored as a node in the graph.

    The value of a variable can be set using `data_manager().set_data(<variable name>, <value>)`. Multiple can be set
    at once using `data_manager().set_data_multiple(<variable name>=<value>, <variable name>=<value>, ...)`.

    Dependencies between variables are added using instance of the `Updater` class, and the function
    `data_manager().add_updater(<updater name>, <updater instance>)`. The `<updater name>` is just a user-defined `str`
    name associated with the updater instance, which is used if the user wishes to remove the updater later on with
    `data_manager().remove_updater(<updater name>)`.

    Each updater defines a dependency of a set of variables on a set of other variables, as well as the mapping between
    them.

    Each variable (node) may be updated by:
    - no updaters, so its value is always valid, and is only ever changed using `data_manager().set_data` /
    `data_manager().set_data_multiple`, or
    - one updater, so its value is invalidated by changes to its dependents, and will be re-evaluated automatically by
    its updater when read.

    By default, all variables are evaluated lazily, i.e. updaters will only be run to re-evaluate invalidated values
    upon reading of the dependent variables. While this is good for optimisation, when data are displayed in a GUI, this
    is generally not desirable, as the user will generally wish to only be shown valid data. In this case, variables
    can manually be set to non-lazy have evaluation using
    `data_manager().set_evaluation_laziness(<variable name>, lazily_evaluated=False)`.

    It is sometimes useful to know when a variable's value changes, e.g. for updating GUI elements. Callbacks can be
    added using `data_manager().add_callback(<variable name>, <callback name>, <callback value>)`. The `<callback name>`
    is just a user-defined `str` name associated with the callback, which is used if the user wishes to remove the
    callback later on with `data_manager().remove_callback(<variable name>, <callback name>)`.
    """

    def __init__(self):
        self.__nodes: dict[str, Node] = dict()
        self.__updaters: dict[str, Updater] = dict()
        self.__in_top_level_call: bool = True
        self.__children: weakref.WeakSet[
            IChildDirectedAcyclicDataGraph] = weakref.WeakSet()  # children add themselves, and will be automatically
        # removed on destruction

    @staticmethod
    def __data_mutating(function):
        """
        A private decorator for marking methods as having the potential to make data dirty, such that
        non-lazily-evaluated nodes can be cleaned automatically. This tracks whether it is in the top-level call to
        avoid repeated evaluation of the same node, which would waste computation.
        :param function:
        :return:
        """

        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            this_is_top_level_call = self.__in_top_level_call
            self.__in_top_level_call = False
            ret = function(self, *args, **kwargs)
            if this_is_top_level_call:
                err = self.__clean_graph()
                if isinstance(err, Error):
                    return Error(f"Error cleaning graph after call to data mutating function: {err.description}.")
            self.__in_top_level_call = this_is_top_level_call
            return ret

        return wrapper

    def __str__(self) -> str:
        ret = "StandaloneDADG(\n"
        ret += "\tNodes:\n"
        for node_name, value in self.__nodes.items():
            ret += f"\t\t{node_name}: {value},\n"
        ret += "\tUpdaters:\n"
        for updater_name, value in self.__updaters.items():
            ret += f"\t\t{updater_name}: {value},\n"
        ret += ")\n"
        return ret

    def has_node(self, node_name: str) -> bool:
        return node_name in self.__nodes

    @__data_mutating
    def get(self, node_name: str, *, soft: bool = False) -> Any | Error:
        """
        Get the data associated with the named node. Will lazily re-calculate the value if previously made dirty by
        changes to other values.
        :param node_name: The name of the node.
        :param soft: [Optional; default=False] Whether to return an Error if there is not enough data to clean the node.
        :return: The data associated with the name node, or an instance of `Error` on failure.
        """
        err = self.__clean_node(node_name, soft=soft)
        if isinstance(err, Error):
            return Error(f"Error cleaning node '{node_name}' before 'get': {err.description}.")
        if self.__nodes[node_name].data is NoNodeData:
            if soft:
                return NoNodeData
            return Error(f"No data stored for node '{node_name}' in graph.")
        return self.__nodes[node_name].data

    @__data_mutating
    def set(self, node_name: str, data: Any, *, check_equality: bool = False) -> None | Error:
        """
        Set the data associated with a named node. Will create the node if it doesn't exist.
        :param node_name: Name of the node.
        :param data: New data to assign.
        :param check_equality: [Optional; default=False] Whether to check the new value against the old value,
        and leave the node clean if the new value is the same.
        """
        # make sure node exists
        node = self.__get_node_ensure_exists(node_name)
        # set the data and make not dirty
        if check_equality and node.data == data:
            return None
        self.__send_children_copies(node_name)
        node.data = data
        node.dirty = False
        # call the node's set callbacks
        for _, callback in node.observers.items():
            err = callback(data)
            if isinstance(err, Error):
                return Error(f"Observer callback returned error: {err.description}.")
        # make all dependents dirty
        for dependent in node.dependents:
            self.__set_dirty(dependent)
        return None

    @__data_mutating
    def set_multiple(self, **kwargs) -> None | Error:
        """
        Set the data associated with multiple named nodes. Will create the node if it doesn't exist.
        Each keyword argument `name=value` will set the data `value` for the node `name`.
        :return: Nothing on success; an `Error` on failure.
        """
        for node_name, data in kwargs.items():
            err = self.set(node_name, data)
            if isinstance(err, Error):
                return Error(f"Error setting multiple data; specifically for '{node_name}': {err.description}.")
        return None

    @__data_mutating
    def set_evaluation_laziness(self, node_name: str, *, lazily_evaluated: bool) -> None:
        """
        Set whether the named node will be evaluated lazily. Nodes will be evaluated lazily by default. A node that is
        not evaluated lazily will be re-evaluated if dirty and the required data exists, every time data in the graph
        changes.
        :param node_name: The name of the node to modify
        :param lazily_evaluated: Whether the named node should be evaluated lazily
        """
        node = self._get_node_if_exists(node_name)
        if node is not None:
            if node.lazily_evaluated == lazily_evaluated:
                return
            self.__send_children_copies(node_name)
        node = self.__get_node_ensure_exists(node_name)
        node.lazily_evaluated = lazily_evaluated

    def observe(self, node_name: str, observer_name: str, callback: Callable[[Any], None]) -> None:
        """
        Add a callback function to a node that will be called whenever the node's data changes.
        :param node_name:
        :param observer_name:
        :param callback:
        :return:
        """
        self.__get_node_ensure_exists(node_name).observers[observer_name] = callback

    def remove_observer(self, node_name: str, observer_name: str) -> None | Error:
        node: Node | None = self._get_node_if_exists(node_name)
        if node is None:
            return Error(f"No node named '{node_name}' from which to remove callback.")
        self.__send_children_copies(node_name)
        if observer_name not in node.observers:
            return Error(f"No observer named '{observer_name}' in node '{node_name}'.")
        node.observers.pop(observer_name)
        return None

    @__data_mutating
    def add_updater(self, updater_name: str, updater: Updater) -> None | Error:
        """
        Add a new updater object to the DAG. Each node may only be updated by a single updater, or no updater.
        :param updater_name: A human-readable name to assign to the added updater.
        :param updater: The updater object. Can be created from a function using the `dag_updater` decorator.
        :return: None, or an `Error` object on failure.
        """
        # insert the new updater, checking that there isn't already one of the same name
        if updater_name in self.__updaters:
            return Error(f"Updater named '{updater_name}' already exists in graph.")
        self.__updaters[updater_name] = updater
        # set the updater names for updated nodes
        for updated in updater.returned:
            self.__send_children_copies(updated)
            node = self.__get_node_ensure_exists(updated)
            if node.updater is None:
                node.updater = updater_name
            else:
                return Error(f"Node {updated} is already updated by {node.updater}; tried to add updater "
                             f"{updater_name} which wants to update the same node.")
        # add dependencies now that updated nodes have assigned updaters
        for dependency in updater.dependencies:
            self.__add_dependency(dependency)
        # check that the graph is not acyclic
        if not self.__check_acyclic():
            self.remove_updater(updater_name)
            return Error(f"Failed to add updater '{updater_name}', as this added cycles to the graph.")
        return None

    def remove_updater(self, updater_name: str) -> None | Error:
        # check updater exists with given updater_name
        if updater_name not in self.__updaters:
            return Error(f"Failed to remove updater '{updater_name}'; it doesn't exist.")
        # remove the dependencies
        for dependency in self.__updaters[updater_name].dependencies:
            self.__remove_dependency(dependency)
        # remove the updater names for the updated nodes
        for updated in self.__updaters[updater_name].returned:
            if updated not in self.__nodes:
                continue
            self.__send_children_copies(updated)
            self.__nodes[updated].updater = None
        # remove the updater
        self.__updaters.pop(updater_name)
        return None

    @__data_mutating
    def get_with_args(self, args: list[FunctionArgument]) -> dict[str, Any] | Error:
        """
        Get the given list of named arguments from the `DAG`, returning an Error if any required argument is not
        available.
        :param args: list of function arguments, marked with whether they are required
        :return: A dictionary mapping argument name to retrieved value.
        """
        ret = {}
        for arg in args:
            value = self.get(arg.name)
            if isinstance(value, Error):
                if arg.required:
                    return Error(f"Error getting argument {arg.name}: '{value.description}'")
                continue
            ret[arg.name] = value
        return ret

    def render(self) -> None:
        import graphviz
        g = graphviz.Digraph(format='png')
        for name, updater in self.__updaters.items():
            for dependency in updater.dependencies:
                g.edge(dependency.depender, dependency.depended)
        g.render(directory=".")

    ##### Protected methods and properties

    def _get_node_if_exists(self, node_name: str) -> Node | None:
        return self.__nodes[node_name] if node_name in self.__nodes else None

    @property
    def _nodes(self) -> dict[str, Node]:
        return self.__nodes

    @property
    def _updaters(self) -> dict[str, Updater]:
        return self.__updaters

    def _add_child(self, child: IChildDirectedAcyclicDataGraph):
        self.__children.add(child)

    ##### Private methods

    def __get_node_ensure_exists(self, node_name: str) -> Node:
        if node_name in self.__nodes:
            return self.__nodes[node_name]
        self.__nodes[node_name] = Node()
        return self.__nodes[node_name]

    def __set_dirty(self, node_name: str) -> None:
        self.__send_children_copies(node_name)
        # make sure node exists
        node: Node = self.__get_node_ensure_exists(node_name)
        # make node and all dependents dirty
        node.dirty = True
        for dependent in node.dependents:
            self.__set_dirty(dependent)

    def __add_dependency(self, dependency: Dependency) -> None:
        # make sure both nodes exist
        self.__get_node_ensure_exists(dependency.depender)
        depended_node: Node = self.__get_node_ensure_exists(dependency.depended)
        # make sure depender is in depended's list of dependents
        if dependency.depender not in depended_node.dependents:
            self.__send_children_copies(dependency.depended)
            depended_node.dependents.append(dependency.depender)
        # make depender dirty
        self.__set_dirty(dependency.depender)

    def __remove_dependency(self, dependency: Dependency) -> None | Error:
        # check both nodes exist
        depender_node: Node | None = self._get_node_if_exists(dependency.depender)
        depended_node: Node | None = self._get_node_if_exists(dependency.depended)
        if depender_node is None:
            return Error(f"Error removing dependency: depender {dependency.depender} doesn't exist.")
        if depended_node is None:
            return Error(f"Error removing dependency: depender {dependency.depended} doesn't exist.")
        # make sure depender is not in depended's list of dependents
        if dependency.depender in depended_node.dependents:
            self.__send_children_copies(dependency.depender)
            depended_node.dependents.remove(dependency.depender)
        return None

    def __run_updater(self, updater_name: str, *, soft: bool = False) -> None | Error:
        # get the updater object
        if updater_name not in self.__updaters:
            return Error(f"No updater named '{updater_name}' in graph.")
        updater = self.__updaters[updater_name]
        # get the arguments for the function
        kwargs = self.get_with_args(updater.arguments)
        if isinstance(kwargs, Error):
            if soft:
                return None
            return Error(f"Failed to get arguments to run updater '{updater_name}': {kwargs.description}")
        # execute the function
        res = updater.function(**kwargs)
        if isinstance(res, Error):
            return Error(f"Error running updater function '{updater_name}': {res.description}")
        # for each promised value, check it exists, and set the data
        if not isinstance(res, dict):
            return Error(f"Expected data updater function '{updater_name}' to return a dictionary.")
        for variable_updated in updater.returned:
            if variable_updated not in res:
                return Error(
                    f"Variable '{variable_updated}' not returned by updater function '{updater_name}' which promised "
                    f"it.")
            err = self.set(variable_updated, res.pop(variable_updated))
            if isinstance(err, Error):
                return Error(f"Error setting value for '{variable_updated}' after running updater '{updater_name}': "
                             f"{err.description}.")
        # check for values returned that weren't promised
        if len(res) > 0:
            variable_names = "', '".join(list(res.keys()))
            return Error(f"Updater function '{updater_name}' returned unexpected variables: '{variable_names}'")
        return None

    def __clean_node(self, node_name: str, *, soft: bool = False) -> None | Error:
        node: Node | None = self._get_node_if_exists(node_name)
        if node is None:
            return Error(f"No node named '{node_name}' in graph.")
        if not node.dirty:
            return None
        if node.updater is None:
            if soft:
                return None
            return Error(f"Dirty node '{node_name}' cannot be cleaned as it has no updater.")
        err = self.__run_updater(node.updater, soft=soft)
        if isinstance(err, Error):
            return Error(f"Failed to clean dirty node '{node_name}'; error running updater '{node.updater}': "
                         f"{err.description}.")
        if node.dirty and not soft:
            return Error(f"Node '{node_name}' still dirty after running updater '{node.updater}'.")
        return None

    def __clean_graph(self) -> None | Error:
        for node_name, node in self.__nodes.items():
            if node.lazily_evaluated:
                continue
            err = self.__clean_node(node_name, soft=True)
            if isinstance(err, Error):
                return Error(f"Graph clean failed on node '{node_name}': {err.description}.")
        return None

    def __eval_in_degrees(self) -> dict[str, int]:
        ret = {node_name: 0 for node_name in self.__nodes}
        for _, node in self.__nodes.items():
            for dependent in node.dependents:
                ret[dependent] += 1
        return ret

    def __check_acyclic(self) -> bool:
        in_degrees: dict[str, int] = self.__eval_in_degrees()
        q = collections.deque([node_name for node_name in self.__nodes if in_degrees[node_name] == 0])
        count: int = 0
        while q:
            first = q.popleft()
            count += 1
            for dependent in self.__nodes[first].dependents:
                in_degrees[dependent] -= 1
                if in_degrees[dependent] == 0:
                    q.append(dependent)
        return count == len(self.__nodes)

    def __send_children_copies(self, node_name: str) -> None:
        for child in self.__children:
            child._make_copy(node_name)


class StandaloneDADGSingleton(SingletonConfigurable):
    _standalone_dadg: StandaloneDADG | None = traitlets.Instance(StandaloneDADG, allow_none=True, default_value=None)

    def get(self, **init_kwargs) -> StandaloneDADG:
        if self._standalone_dadg is None:
            self._standalone_dadg = StandaloneDADG(**init_kwargs)
            logger.info(f"Data manager initialised with the following parameters: {init_kwargs}")
        return self._standalone_dadg
