import collections
import copy
import functools
import logging
from typing import Any, Callable, NamedTuple
import traitlets
from traitlets.config import SingletonConfigurable
import weakref

from reg23_experiments.data.structs import Error
from reg23_experiments.utils.reflection import FunctionArgument

__all__ = ["Dependency", "NoNodeDataType", "NoNodeData", "Updater", "DAG", "ChildDAG", "DataManagerSingleton"]

logger = logging.getLogger(__name__)


class Dependency(NamedTuple):
    depender: str
    depended: str


class NoNodeDataType:
    pass


NoNodeData = NoNodeDataType()


class Node(traitlets.HasTraits):
    dependents: list[str] = traitlets.List(trait=traitlets.Unicode(), default_value=[])
    dirty: bool = traitlets.Bool(default_value=False)
    updater: str = traitlets.Unicode(allow_none=True, default_value=None)
    lazily_evaluated: bool = traitlets.Bool(default_value=True)
    data: Any = traitlets.Any(allow_none=True, default_value=NoNodeData)
    set_callbacks: dict[str, Callable] = traitlets.Dict(key_trait=traitlets.Unicode(), value_trait=traitlets.Callable())

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


class DAG:
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
        self._nodes: dict[str, Node] = dict()
        self._updaters: dict[str, Updater] = dict()
        self._in_top_level_call: bool = True
        self._children = weakref.WeakSet()  # children add themselves, and will be automatically removed on destruction

    @staticmethod
    def _data_mutating(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            this_is_top_level_call = self._in_top_level_call
            self._in_top_level_call = False
            ret = function(self, *args, **kwargs)
            if this_is_top_level_call:
                err = self._clean_graph()
                if isinstance(err, Error):
                    return Error(f"Error cleaning graph after call to data mutating function: {err.description}.")
            self._in_top_level_call = this_is_top_level_call
            return ret

        return wrapper

    def __str__(self) -> str:
        ret = "DataDAG(\n"
        ret += "\tNodes:\n"
        for name, value in self._nodes.items():
            ret += f"\t\t{name}: {value},\n"
        ret += "\tUpdaters:\n"
        for name, value in self._updaters.items():
            ret += f"\t\t{name}: {value},\n"
        ret += ")\n"
        return ret

    @_data_mutating
    def get(self, name: str, *, soft: bool = False) -> Any | Error:
        """
        Get the data associated with the named node. Will lazily re-calculate the value if previously made dirty by
        changes to other values.
        :param name: The name of the node.
        :param soft: [Optional; default=False] Whether to return an Error if there is not enough data to clean the node.
        :return: The data associated with the name node, or an instance of `Error` on failure.
        """
        err = self._clean_node(name, soft=soft)
        if isinstance(err, Error):
            return Error(f"Error cleaning node '{name}' before 'get': {err.description}.")
        if self._nodes[name].data is NoNodeData:
            if soft:
                return NoNodeData
            return Error(f"No data stored for node '{name}' in graph.")
        return self._nodes[name].data

    @_data_mutating
    def add_node(self, name: str, *, data: Any = NoNodeData) -> None:
        """
        Create the named node if it doesn't already exist, and optionally assign data to it.
        :param name: Name of the node.
        :param data: [Optional] Data to assign. No assignment will be made if 'NoNodeData' is passed.
        """
        if name not in self._nodes:
            self._nodes[name] = Node()
        if data is not NoNodeData:
            self.set_data(name, data)

    @_data_mutating
    def set_evaluation_laziness(self, node_name: str, *, lazily_evaluated: bool) -> None:
        """
        Set whether the named node will be evaluated lazily. Nodes will be evaluated lazily by default. A node that is
        not evaluated lazily will be re-evaluated if dirty and the required data exists, every time data in the graph
        changes.
        :param node_name: The name of the node to modify
        :param lazily_evaluated: Whether the named node should be evaluated lazily
        """
        self._send_children_copies(node_name)
        self.add_node(node_name)
        self._nodes[node_name].lazily_evaluated = lazily_evaluated

    @_data_mutating
    def set_data(self, node_name: str, data: Any, *, check_equality: bool = False) -> None | Error:
        """
        Set the data associated with a named node. Will create the node if it doesn't exist.
        :param node_name: Name of the node.
        :param data: New data to assign.
        :param check_equality: [Optional; default=False] Whether to check the new value against the old value,
        and leave the node clean if the new value is the same.
        """
        # make sure node exists
        self.add_node(node_name)
        # set the data and make not dirty
        if check_equality and self._nodes[node_name].data == data:
            return None
        self._send_children_copies(node_name)
        self._nodes[node_name].data = data
        self._nodes[node_name].dirty = False
        # call the node's set callbacks
        for _, callback in self._nodes[node_name].set_callbacks.items():
            err = callback(data)
            if isinstance(err, Error):
                return Error(f"Set callback returned error: {err.description}.")
        # make all dependents dirty
        for dependent in self._nodes[node_name].dependents:
            self._set_dirty(dependent)
        return None

    @_data_mutating
    def set_data_multiple(self, **kwargs) -> None | Error:
        """
        Set the data associated with multiple named nodes. Will create the node if it doesn't exist.
        Each keyword argument `name=value` will set the data `value` for the node `name`.
        :return: Nothing on success; an `Error` on failure.
        """
        for name, data in kwargs.items():
            err = self.set_data(name, data)
            if isinstance(err, Error):
                return Error(f"Error setting multiple data; specifically for '{name}': {err.description}.")
        return None

    def add_callback(self, node_name: str, callback_name: str, callback: Callable[[Any], None]) -> None:
        """
        Add a callback function to a node that will be called whenever the node's data changes.
        :param node_name:
        :param callback_name:
        :param callback:
        :return:
        """
        self._send_children_copies(node_name)
        self.add_node(node_name)
        self._nodes[node_name].set_callbacks[callback_name] = callback

    def remove_callback(self, node_name: str, callback_name: str) -> None | Error:
        if node_name not in self._nodes:
            return Error(f"No node named '{node_name}' from which to remove callback.")
        self._send_children_copies(node_name)
        if callback_name not in self._nodes[node_name].set_callbacks:
            return Error(f"No callback named '{callback_name}' in node '{node_name}'.")
        self._nodes[node_name].set_callbacks.pop(callback_name)
        return None

    @_data_mutating
    def add_updater(self, name: str, updater: Updater) -> None | Error:
        """
        Add a new updater object to the DAG. Each node may only be updated by a single updater, or no updater.
        :param name: A human-readable name to assign to the added updater.
        :param updater: The updater object. Can be created from a function using the `dag_updater` decorator.
        :return: None, or an `Error` object on failure.
        """
        # insert the new updater, checking that there isn't already one of the same name
        if name in self._updaters:
            return Error(f"Updater named '{name}' already exists in graph.")
        self._updaters[name] = updater
        # set the updater names for updated nodes
        for updated in updater.returned:
            self._send_children_copies(updated)
            self.add_node(updated)
            if self._nodes[updated].updater is None:
                self._nodes[updated].updater = name
            else:
                return Error(
                    f"Node {updated} is already updated by {self._nodes[updated].updater}; tried to add updater {name} "
                    f"which wants to update the same node.")
        # add dependencies now that updated nodes have assigned updaters
        for dependency in updater.dependencies:
            self._add_dependency(dependency)
        # check that the graph is not acyclic
        if not self._check_acyclic():
            self.remove_updater(name)
            return Error(f"Failed to add updater '{name}', as this added cycles to the graph.")
        return None

    def remove_updater(self, name: str) -> None | Error:
        # check updater exists with given name
        if name not in self._updaters:
            return Error(f"Failed to remove updater '{name}'; it doesn't exist.")
        # remove the dependencies
        for dependency in self._updaters[name].dependencies:
            self._remove_dependency(dependency)
        # remove the updater names for the updated nodes
        for updated in self._updaters[name].returned:
            if updated not in self._nodes:
                continue
            self._send_children_copies(updated)
            self._nodes[updated].updater = None
        # remove the updater
        self._updaters.pop(name)
        return None

    @_data_mutating
    def get_args(self, args: list[FunctionArgument]) -> dict[str, Any] | Error:
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
        for name, updater in self._updaters.items():
            for dependency in updater.dependencies:
                g.edge(dependency.depender, dependency.depended)
        g.render(directory=".")

    def node_exists(self, name: str) -> bool:
        return name in self._nodes

    def _get_node_if_exists(self, name: str) -> Node | None:
        return self._nodes[name] if name in self._nodes else None

    def _set_dirty(self, name: str) -> None:
        # make sure node exists
        self.add_node(name)
        # make node and all dependents dirty
        self._send_children_copies(name)
        self._nodes[name].dirty = True
        for dependent in self._nodes[name].dependents:
            self._set_dirty(dependent)

    def _add_dependency(self, dependency: Dependency) -> None:
        # make sure both nodes exist
        self.add_node(dependency.depender)
        self.add_node(dependency.depended)
        # make sure depender is in depended's list of dependents
        if dependency.depender not in self._nodes[dependency.depended].dependents:
            self._send_children_copies(dependency.depended)
            self._nodes[dependency.depended].dependents.append(dependency.depender)
        # make depender dirty
        self._send_children_copies(dependency.depender)
        self._nodes[dependency.depender].dirty = True

    def _remove_dependency(self, dependency: Dependency) -> None | Error:
        # check both nodes exist
        if dependency.depender not in self._nodes:
            return Error(f"Error removing dependency: depender {dependency.depender} doesn't exist.")
        if dependency.depended not in self._nodes:
            return Error(f"Error removing dependency: depender {dependency.depended} doesn't exist.")
        # make sure depender is not in depended's list of dependents
        if dependency.depender in self._nodes[dependency.depended].dependents:
            self._send_children_copies(dependency.depender)
            self._nodes[dependency.depended].dependents.remove(dependency.depender)
        return None

    def _run_updater(self, name: str, *, soft: bool = False) -> None | Error:
        # get the updater object
        if name not in self._updaters:
            return Error(f"No updater named '{name}' in graph.")
        updater = self._updaters[name]
        # get the arguments for the function
        args = self.get_args(updater.arguments)
        if isinstance(args, Error):
            if soft:
                return None
            return Error(f"Failed to get arguments to run updater '{name}': {args.description}")
        # execute the function
        res = updater.function(**args)
        if isinstance(res, Error):
            return Error(f"Error running updater function '{name}': {res.description}")
        # for each promised value, check it exists, and set the data
        if not isinstance(res, dict):
            return Error(f"Expected data updater function '{name}' to return a dictionary.")
        for variable_updated in updater.returned:
            if variable_updated not in res:
                return Error(
                    f"Variable '{variable_updated}' not returned by updater function '{name}' which promised it.")
            err = self.set_data(variable_updated, res.pop(variable_updated))
            if isinstance(err, Error):
                return Error(
                    f"Error setting value for '{variable_updated}' after running updater '{name}': {err.description}.")
        # check for values returned that weren't promised
        if len(res) > 0:
            variable_names = "', '".join(list(res.keys()))
            return Error(f"Updater function '{name}' returned unexpected variables: '{variable_names}'")
        return None

    def _clean_node(self, name: str, *, soft: bool = False) -> None | Error:
        if name not in self._nodes:
            return Error(f"No node named '{name}' in graph.")
        if not self._nodes[name].dirty:
            return None
        if self._nodes[name].updater is None:
            if soft:
                return None
            return Error(f"Dirty node '{name}' cannot be cleaned as it has no updater.")
        err = self._run_updater(self._nodes[name].updater, soft=soft)
        if isinstance(err, Error):
            return Error(f"Failed to clean dirty node '{name}'; error running updater '{self._nodes[name].updater}': "
                         f"{err.description}.")
        if self._nodes[name].dirty and not soft:
            return Error(f"Node '{name}' still dirty after running updater '{self._nodes[name].updater}'.")
        return None

    def _clean_graph(self) -> None | Error:
        for name, node in self._nodes.items():
            if node.lazily_evaluated:
                continue
            err = self._clean_node(name, soft=True)
            if isinstance(err, Error):
                return Error(f"Graph clean failed on node '{name}': {err.description}.")
        return None

    def _eval_in_degrees(self) -> dict[str, int]:
        ret = {node_name: 0 for node_name in self._nodes}
        for name, node in self._nodes.items():
            for dependent in node.dependents:
                ret[dependent] += 1
        return ret

    def _check_acyclic(self) -> bool:
        in_degrees: dict[str, int] = self._eval_in_degrees()
        q = collections.deque([node_name for node_name in self._nodes if in_degrees[node_name] == 0])
        count: int = 0
        while q:
            first = q.popleft()
            count += 1
            for dependent in self._nodes[first].dependents:
                in_degrees[dependent] -= 1
                if in_degrees[dependent] == 0:
                    q.append(dependent)
        return count == len(self._nodes)

    def _send_children_copies(self, node_name: str) -> None:
        for child in self._children:
            child.make_copy(node_name)


class ChildDAG:
    def __init__(self, parent: DAG):
        self._parent = parent  # strong reference to parent, so parent stays alive as long as there are children
        # parent holds weak reference to this, which will automatically be removed from the WeakSet when this is
        # destroyed
        self._nodes_inheriting: list[str] = list(self._parent._nodes.keys())
        self._parent._children.add(self)
        self._nodes: dict[str, Node] = dict()
        # ToDo?: child's own updaters?
        self._in_top_level_call: bool = True

    @staticmethod
    def _data_mutating(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            this_is_top_level_call = self._in_top_level_call
            self._in_top_level_call = False
            ret = function(self, *args, **kwargs)
            if this_is_top_level_call:
                err = self._clean_graph()
                if isinstance(err, Error):
                    return Error(f"Error cleaning graph after call to data mutating function: {err.description}.")
            self._in_top_level_call = this_is_top_level_call
            return ret

        return wrapper

    def make_copy(self, node_name: str) -> None | Error:
        if node_name in self._nodes or node_name not in self._nodes_inheriting:
            return None
        maybe_parent_node = self._parent._get_node_if_exists(node_name)
        if isinstance(maybe_parent_node, Error):
            return Error(f"Could not copy node '{node_name}' to child as it doesn't exist in the parent.")
        self._nodes[node_name] = copy.deepcopy(maybe_parent_node)
        return None

    @_data_mutating
    def get(self, name: str, *, soft: bool = False) -> Any | Error:
        """
        Get the data associated with the named node. Will lazily re-calculate the value if previously made dirty by
        changes to other values.
        :param name: The name of the node.
        :param soft: [Optional; default=False] Whether to return an Error if there is not enough data to clean the node.
        :return: The data associated with the name node, or an instance of `Error` on failure.
        """
        err = self._clean_node(name, soft=soft)
        if isinstance(err, Error):
            return Error(f"Error cleaning node '{name}' before 'get': {err.description}.")
        data: Any = self._get_node_if_exists(name).data
        if data is NoNodeData:
            if soft:
                return NoNodeData
            return Error(f"No data stored for node '{name}' in graph.")
        return data

    @_data_mutating
    def add_node(self, name: str, *, data: Any = NoNodeData) -> None:
        """
        Create the named node if it doesn't already exist, and optionally assign data to it.
        :param name: Name of the node.
        :param data: [Optional] Data to assign. No assignment will be made if 'NoNodeData' is passed.
        """
        if not self.node_exists(name):
            self._nodes[name] = Node()
        if data is not NoNodeData:
            self.set_data(name, data)

    @_data_mutating
    def set_evaluation_laziness(self, node_name: str, *, lazily_evaluated: bool) -> None:
        """
        Set whether the named node will be evaluated lazily. Nodes will be evaluated lazily by default. A node that is
        not evaluated lazily will be re-evaluated if dirty and the required data exists, every time data in the graph
        changes.
        :param node_name: The name of the node to modify
        :param lazily_evaluated: Whether the named node should be evaluated lazily
        """
        node = self._get_node_ensure_exists(node_name)
        if node_name not in self._nodes:
            # This is a parent node, so we copy in order to mutate
            self._nodes[node_name] = copy.deepcopy(node)
            node = self._nodes[node_name]
        node.lazily_evaluated = lazily_evaluated

    @_data_mutating
    def set_data(self, node_name: str, data: Any, *, check_equality: bool = False) -> None | Error:
        """
        Set the data associated with a named node. Will create the node if it doesn't exist.
        :param node_name: Name of the node.
        :param data: New data to assign.
        :param check_equality: [Optional; default=False] Whether to check the new value against the old value,
        and leave the node clean if the new value is the same.
        """
        # make sure node exists
        node = self._get_node_ensure_exists(node_name)
        # set the data and make not dirty
        if check_equality and node.data == data:
            return None
        if node_name in self._nodes:
            # this node is already in the child, so we can just mutate it
            node.data = data
            node.dirty = False
        else:
            # this is a parent node, so now we copy this node to the child so that we can mutate it
            self._nodes[node_name] = Node(  #
                dependents=copy.deepcopy(node.dependents),  #
                dirty=False,  #
                updater=copy.deepcopy(node.updater),  #
                lazily_evaluated=node.lazily_evaluated,  #
                data=data,  #
                set_callbacks=copy.deepcopy(node.set_callbacks),  #
            )
            node = self._nodes[node_name]
        # call the node's set callbacks
        for _, callback in node.set_callbacks.items():
            err = callback(data)
            if isinstance(err, Error):
                return Error(f"Set callback returned error: {err.description}.")
        # make all dependents dirty
        for dependent in node.dependents:
            self._set_dirty(dependent)
        return None

    @_data_mutating
    def set_data_multiple(self, **kwargs) -> None | Error:
        """
        Set the data associated with multiple named nodes. Will create the node if it doesn't exist.
        Each keyword argument `name=value` will set the data `value` for the node `name`.
        :return: Nothing on success; an `Error` on failure.
        """
        for name, data in kwargs.items():
            err = self.set_data(name, data)
            if isinstance(err, Error):
                return Error(f"Error setting multiple data; specifically for '{name}': {err.description}.")
        return None

    def add_callback(self, node_name: str, callback_name: str, callback: Callable[[Any], None]) -> None:
        """
        Add a callback function to a node that will be called whenever the node's data changes.
        :param node_name:
        :param callback_name:
        :param callback:
        :return:
        """
        node = self._get_node_ensure_exists(node_name)
        if node_name not in self._nodes:
            # This is a parent node, so we copy in order to mutate
            self._nodes[node_name] = copy.deepcopy(node)
            node = self._nodes[node_name]

        node.set_callbacks[callback_name] = callback

    def remove_callback(self, node_name: str, callback_name: str) -> None | Error:
        node: Node | None = self._get_node_if_exists(node_name)
        if node is None:
            return Error(f"No node named '{node_name}' from which to remove callback.")
        if callback_name not in node.set_callbacks:
            return Error(f"No callback named '{callback_name}' in node '{node_name}'.")
        if node_name not in self._nodes:
            # This is a parent node, so we copy in order to mutate
            self._nodes[node_name] = copy.deepcopy(node)
            node = self._nodes[node_name]
        node.set_callbacks.pop(callback_name)
        return None

    @_data_mutating
    def get_args(self, args: list[FunctionArgument]) -> dict[str, Any] | Error:
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
        for name, updater in self._parent._updaters.items():
            for dependency in updater.dependencies:
                g.edge(dependency.depender, dependency.depended)
        g.render(directory=".")

    def node_exists(self, name: str) -> bool:
        if name in self._nodes:
            return True
        return self._parent.node_exists(name)

    def _get_node_if_exists(self, name: str) -> Node | None:
        if name in self._nodes:
            return self._nodes[name]
        return self._parent._get_node_if_exists(name)

    def _get_node_ensure_exists(self, name: str) -> Node:
        if name in self._nodes:
            return self._nodes[name]
        maybe_parent_node: Node | None = self._parent._get_node_if_exists(name)
        if isinstance(maybe_parent_node, Node):
            return maybe_parent_node
        self._nodes[name] = Node()
        return self._nodes[name]

    def _get_all_name_node_pairs(self) -> list[tuple[str, Node]]:
        return list(self._nodes.items()) + [(name, node) for name, node in self._parent._nodes.items() if
                                            name not in self._nodes]

    def _set_dirty(self, name: str) -> None:
        node = self._get_node_ensure_exists(name)
        if name not in self._nodes:
            # This is a parent node, so we copy in order to mutate
            self._nodes[name] = copy.deepcopy(node)
            node = self._nodes[name]
        # make node and all dependents dirty
        node.dirty = True
        for dependent in node.dependents:
            self._set_dirty(dependent)

    def _run_updater(self, name: str, *, soft: bool = False) -> None | Error:
        # get the updater object
        if name not in self._parent._updaters:
            return Error(f"No updater named '{name}' in graph.")
        updater = self._parent._updaters[name]
        # get the arguments for the function
        args = self.get_args(updater.arguments)
        if isinstance(args, Error):
            if soft:
                return None
            return Error(f"Failed to get arguments to run updater '{name}': {args.description}")
        # execute the function
        res = updater.function(**args)
        if isinstance(res, Error):
            return Error(f"Error running updater function '{name}': {res.description}")
        # for each promised value, check it exists, and set the data
        if not isinstance(res, dict):
            return Error(f"Expected data updater function '{name}' to return a dictionary.")
        for variable_updated in updater.returned:
            if variable_updated not in res:
                return Error(
                    f"Variable '{variable_updated}' not returned by updater function '{name}' which promised it.")
            err = self.set_data(variable_updated, res.pop(variable_updated))
            if isinstance(err, Error):
                return Error(
                    f"Error setting value for '{variable_updated}' after running updater '{name}': {err.description}.")
        # check for values returned that weren't promised
        if len(res) > 0:
            variable_names = "', '".join(list(res.keys()))
            return Error(f"Updater function '{name}' returned unexpected variables: '{variable_names}'")
        return None

    def _clean_node(self, name: str, *, soft: bool = False) -> None | Error:
        node = self._get_node_if_exists(name)
        if node is None:
            return Error(f"No node named '{name}' in graph.")
        if not node.dirty:
            return None
        if node.updater is None:
            if soft:
                return None
            return Error(f"Dirty node '{name}' cannot be cleaned as it has no updater.")
        err = self._run_updater(node.updater, soft=soft)
        if isinstance(err, Error):
            return Error(f"Failed to clean dirty node '{name}'; error running updater '{node.updater}': "
                         f"{err.description}.")
        if node.dirty and not soft:
            return Error(f"Node '{name}' still dirty after running updater '{node.updater}'.")
        return None

    def _clean_graph(self) -> None | Error:
        for name, node in self._get_all_name_node_pairs():
            if node.lazily_evaluated:
                continue
            err = self._clean_node(name, soft=True)
            if isinstance(err, Error):
                return Error(f"Graph clean failed on node '{name}': {err.description}.")
        return None

    def _eval_in_degrees(self) -> dict[str, int]:
        ret = {node_name: 0 for node_name in self._nodes} | {node_name: 0 for node_name in self._parent._nodes if
                                                             node_name not in self._nodes}
        for name, node in self._get_all_name_node_pairs():
            for dependent in node.dependents:
                ret[dependent] += 1
        return ret

    def _check_acyclic(self) -> bool:
        in_degrees: dict[str, int] = self._eval_in_degrees()
        q = collections.deque(
            [node_name for node_name in self._nodes if in_degrees[node_name] == 0] + [node_name for node_name in
                                                                                      self._parent._nodes if
                                                                                      node_name not in self._nodes and
                                                                                      in_degrees[node_name] == 0])
        count: int = 0
        while q:
            first = q.popleft()
            count += 1
            for dependent in self._get_node_if_exists(first).dependents:
                in_degrees[dependent] -= 1
                if in_degrees[dependent] == 0:
                    q.append(dependent)
        return count == len(self._nodes)


class DataManagerSingleton(SingletonConfigurable):
    _data_manager = traitlets.Instance(DAG, allow_none=True, default_value=None)

    def get(self, **init_kwargs) -> DAG:
        if self._data_manager is None:
            self._data_manager = DAG(**init_kwargs)
            logger.info(f"Data manager initialised with the following parameters: {init_kwargs}")
        return self._data_manager
