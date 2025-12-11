import inspect
import traitlets
from traitlets.config import SingletonConfigurable
import functools
from typing import Any, Callable, NamedTuple

from program.lib.structs import Error


class Dependency(NamedTuple):
    depender: str
    depended: str


class NoNodeDataType: pass


NoNodeData = NoNodeDataType()


class Node(traitlets.HasTraits):
    dependents = traitlets.List(trait=traitlets.Unicode(), default_value=[])
    dirty = traitlets.Bool(default_value=False)
    updater = traitlets.Unicode(allow_none=True, default_value=None)
    data = traitlets.Any(allow_none=True, default_value=NoNodeData)
    set_callbacks = traitlets.Dict(key_trait=traitlets.Unicode(), value_trait=traitlets.Callable())

    def __str__(self) -> str:
        ret = "Node(\n"
        ret += "\n\t".join(f"{name}={getattr(self, name)!r}" for name in self.trait_names() if not name.startswith('_'))
        ret += "\n)\n"
        return ret

    @traitlets.validate("dirty")
    def _valid_dirty(self, proposal):
        if self.updater is None and proposal["value"] == True:
            raise traitlets.TraitError("A node with no updater cannot be dirty.")
        return proposal["value"]


class FunctionArgument(NamedTuple):
    name: str
    required: bool


def get_function_arguments(function: Callable) -> list[FunctionArgument]:
    signature = inspect.signature(function)
    ret = []
    for name, param in signature.parameters.items():
        required = (param.default is inspect.Parameter.empty  # no default value
                    and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD))
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            ret.append(FunctionArgument(name=name, required=required))
    return ret


class Updater(NamedTuple):
    function: Callable
    returned: list[str]
    arguments: list[FunctionArgument]
    dependencies: list[Dependency]

    @staticmethod
    def build(*, function: Callable, returned: list[str]) -> 'Updater':
        arguments = get_function_arguments(function)
        dependencies: list[Dependency] = []
        for depended in arguments:
            for depender in returned:
                dependencies.append(Dependency(depender=depender, depended=depended.name))
        return Updater(function=function, returned=returned, arguments=arguments, dependencies=dependencies)


class DAG:
    def __init__(self, lazy: bool = True):
        self._nodes: dict[str, Node] = dict()
        self._updaters: dict[str, Updater] = dict()
        self._lazy = lazy
        self._in_top_level_call: bool = True

    @property
    def lazy(self) -> bool:
        return self._lazy

    @lazy.setter
    def lazy(self, new_value: bool) -> None:
        self._lazy = new_value

    @staticmethod
    def _data_mutating(function):
        @functools.wraps(function)
        def wrapper(self, *args, **kwargs):
            this_is_top_level_call = self._in_top_level_call
            self._in_top_level_call = False
            ret = function(self, *args, **kwargs)
            if this_is_top_level_call and not self.lazy:
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
    def set_data(self, node_name: str, data: Any, *, check_equality: bool = False) -> None | Error:
        """
        Set the data associated with a named node. Will create the node if it doesn't exist.
        :param node_name: Name of the node.
        :param data: New data to assign.
        :param check_equality: [Optional; default=False] Whether to check the new value against the old value, and leave
        the node clean if the new value is the same.
        """
        # make sure node exists
        self.add_node(node_name)
        # set the data and make not dirty
        if check_equality and self._nodes[node_name].data == data:
            return None
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

    def add_callback(self, node_name: str, callback_name: str, callback: Callable[[Any], None]) -> None:
        """
        Add a callback function to a node that will be called whenever the node's data changes.
        :param node_name:
        :param callback_name:
        :param callback:
        :return:
        """
        self.add_node(node_name)
        self._nodes[node_name].set_callbacks[callback_name] = callback

    def remove_callback(self, node_name: str, callback_name: str) -> None | Error:
        if node_name not in self._nodes:
            return Error(f"No node named '{node_name}' from which to remove callback.")
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
            self.add_node(updated)
            if self._nodes[updated].updater is None:
                self._nodes[updated].updater = name
            else:
                return Error(
                    f"Node {updated} is already updated by {self._nodes[updated].updater}; tried to add updater {name} "
                    f"which wants to update the same node.")
        # add dependencies now that updated nodes have assigned updaters
        self._add_dependencies(updater.dependencies)
        return None

    def _set_dirty(self, name: str) -> None:
        # make sure node exists
        self.add_node(name)
        # make node and all dependents dirty
        self._nodes[name].dirty = True
        for dependent in self._nodes[name].dependents:
            self._set_dirty(dependent)

    def _add_dependency(self, dependency: Dependency) -> None:
        # make sure both nodes exist
        self.add_node(dependency.depender)
        self.add_node(dependency.depended)
        # make sure depender is in depended's list of dependents
        if dependency.depender not in self._nodes[dependency.depended].dependents:
            self._nodes[dependency.depended].dependents.append(dependency.depender)
        # make depender dirty
        self._nodes[dependency.depender].dirty = True

    def _add_dependencies(self, dependencies: list[Dependency]) -> None:
        for dependency in dependencies:
            self._add_dependency(dependency)

    def _get_args(self, args: list[FunctionArgument]) -> dict[str, Any] | Error:
        ret = {}
        for arg in args:
            value = self.get(arg.name)
            if isinstance(value, Error):
                if arg.required:
                    return Error(f"Error getting argument {arg.name}: '{value.description}'")
                continue
            ret[arg.name] = value
        return ret

    def _run_updater(self, name: str, *, soft: bool = False) -> None | Error:
        # get the updater object
        if name not in self._updaters:
            return Error(f"No updater named '{name}' in graph.")
        updater = self._updaters[name]
        # get the arguments for the function
        args = self._get_args(updater.arguments)
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
        for name in self._nodes:
            err = self._clean_node(name, soft=True)
            if isinstance(err, Error):
                return Error(f"Graph clean failed on node '{name}': {err.description}.")
        return None


def dag_updater(*, names_returned: list[str]):
    """
    A decorator for turning a function into an `Updater` for use in a `DAG`. The named arguments of the function will be
    interpreted as nodes from which to read data from the `DAG`. The return value of the function must be a dictionary
    mapping string 'names' to values, where the strings must match the contents of the list `names_returned`. The
    returned names will be used as node names for updating values in the `DAG` with the returned values.
    :param names_returned: A list of the names of the values returned by the function.
    :return: An instance of `Updater`, built using the decorated function.
    """

    def decorator(function):
        return Updater.build(function=function, returned=names_returned)

    return decorator


@dag_updater(names_returned=["similarity"])
def try_updater(fixed_image: float, moving_image: float) -> dict[str, Any]:
    return {"similarity": fixed_image * moving_image}


class DataManagerSingleton(SingletonConfigurable):
    _data_manager = traitlets.Instance(DAG, allow_none=True, default_value=None)

    def get(self, **init_kwargs) -> DAG:
        if self._data_manager is None:
            self._data_manager = DAG(**init_kwargs)
            print(f"Data manager initialised with the following parameters: {init_kwargs}")
        return self._data_manager


def init_data_manager(**kwargs) -> DAG:
    return DataManagerSingleton.instance().get(**kwargs)


def data_manager() -> DAG:
    return DataManagerSingleton.instance().get()


def main():
    data_manager = DAG(lazy=False)

    err = data_manager.add_updater("similarity_metric", try_updater)
    if isinstance(err, Error):
        print(err)
        return

    data_manager.set_data("fixed_image", 2.0)
    data_manager.set_data("moving_image", 3.0)

    # print(f"Data manager:\n{data_manager}")

    print(data_manager.get("similarity"))


if __name__ == "__main__":
    main()
