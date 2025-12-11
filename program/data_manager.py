import inspect
import traitlets
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
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            ret.append(FunctionArgument(name=name, required=param is inspect._empty))
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
    def __init__(self):
        self._nodes: dict[str, Node] = dict()
        self._updaters: dict[str, Updater] = dict()

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

    def get(self, name: str) -> Any | Error:
        """
        Get the data associated with the named node. Will lazily re-calculate the value if previously made dirty by
        changes to other values.
        :param name: The name of the node.
        :return: The data associated with the name node, or an instance of `Error` on failure.
        """
        if name not in self._nodes:
            return Error(f"No node named '{name}' in graph.")
        if self._nodes[name].dirty:
            err = self._run_updater(self._nodes[name].updater)
            if isinstance(err, Error):
                return Error(f"Node '{name}' is dirty on get, error running updater '{self._nodes[name].updater}': "
                             f"{err.description}.")
        if self._nodes[name].dirty:
            return Error(f"Node '{name}' still dirty after running updater '{self._nodes[name].updater}'.")
        if self._nodes[name].data is NoNodeData:
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

    def set_data(self, node_name: str, data: Any) -> None:
        """
        Set the data associated with a named node. Will create the node if it doesn't exist.
        :param node_name: Name of the node.
        :param data: New data to assign.
        """
        # make sure node exists
        self.add_node(node_name)
        # set the data and make not dirty
        self._nodes[node_name].data = data
        self._nodes[node_name].dirty = False
        # make all dependents dirty
        for dependent in self._nodes[node_name].dependents:
            self._set_dirty(dependent)

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

    def _run_updater(self, name: str) -> None | Error:
        # get the updater object
        if name not in self._updaters:
            return Error(f"No updater named '{name}' in graph.")
        updater = self._updaters[name]
        # get the arguments for the function
        args = self._get_args(updater.arguments)
        if isinstance(args, Error):
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
            self.set_data(variable_updated, res.pop(variable_updated))
        # check for values returned that weren't promised
        if len(res) > 0:
            variable_names = "', '".join(list(res.keys()))
            return Error(f"Updater function '{name}' returned unexpected variables: '{variable_names}'")
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


def main():
    data_manager = DAG()

    err = data_manager.add_updater("similarity_metric", try_updater)
    if isinstance(err, Error):
        print(err)
        return

    data_manager.set_data("fixed_image", 2.0)
    data_manager.set_data("moving_image", 3.0)

    print(f"Data manager:\n{data_manager}")

    print(data_manager.get("similarity"))


if __name__ == "__main__":
    main()
