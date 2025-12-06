from typing import NamedTuple, Any, Callable, Generic, TypeVar
import inspect

from program.lib.structs import Error

class Dependency(NamedTuple):
    depender: str
    depended: str


def call_function_with_arg_getter(function: Callable, arg_getter: Callable[[str], Any | Error]) -> Any | Error:
    signature = inspect.signature(function)
    bound = {}

    for name, param in signature.parameters.items():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            arg = arg_getter(name)
            if isinstance(arg, Error):
                return Error(f"Error getting argument {name}: '{arg.description}'")
            if arg is not None:
                bound[name] = arg
            elif param.default is inspect._empty:
                return Error(f"Missing required argument: '{name}'")

    return function(**bound)


T = TypeVar("T")


class NodeValue(Generic[T]):
    def __init__(self, data: T | None, depends_on: list[str], depended_on_by: list[str]):
        self.data = data
        self.depends_on = depends_on
        self.depended_on_by = depended_on_by

    @staticmethod
    def default() -> 'NodeValue':
        return NodeValue(None, [], [])


class TwoWayDependencyGraph(Generic[T]):
    def __init__(self):
        self._nodes: dict[str, NodeValue[T]] = dict()

    def get_node(self, node_name: str) -> NodeValue[T] | None:
        if node_name in self._nodes:
            return self._nodes[node_name]
        else:
            return None

    def set_data(self, node_name: str, data: Any) -> None:
        if node_name in self._nodes:
            self._nodes[node_name] = NodeValue[T].default()
        self._nodes[node_name].data = data

    def add_node(self, name: str, *, depends_on: list[str] = None, depended_on_by: list[str] = None,
                 data: Any = None) -> None:
        if name not in self._nodes:
            self._nodes[name] = NodeValue[T].default()
        self._nodes[name].data = data
        for node in depends_on:
            if node not in self._nodes[name].depends_on:
                self._nodes[name].depends_on.append(node)
            if node not in self._nodes:
                self._nodes[node] = NodeValue[T].default()
            if name not in self._nodes[node].depended_on_by:
                self._nodes[node].depended_on_by.append(name)
        for node in depended_on_by:
            if node not in self._nodes[name].depended_on_by:
                self._nodes[name].depended_on_by.append(node)
            if node not in self._nodes:
                self._nodes[node] = NodeValue[T].default()
            if name not in self._nodes[node].depends_on:
                self._nodes[node].depends_on.append(name)

    def add_dependency(self, dependency: Dependency) -> None:
        if dependency.depender not in self._nodes:
            self._nodes[dependency.depender] = NodeValue[T].default()
        if dependency.depended not in self._nodes[dependency.depender].depends_on:
            self._nodes[dependency.depender].depends_on.append(dependency.depended)
        if dependency.depended not in self._nodes:
            self._nodes[dependency.depended] = NodeValue[T].default()
        if dependency.depender not in self._nodes[dependency.depended].depended_on_by:
            self._nodes[dependency.depended].depended_on_by.append(dependency.depender)

    def add_dependencies(self, dependencies: list[Dependency]) -> None:
        for dependency in dependencies:
            self.add_dependency(dependency)


class FunctionArgument(NamedTuple):
    name: str
    required: bool


def get_function_arguments(function: Callable) -> list[FunctionArgument]:
    signature = inspect.signature(function)
    ret = []
    for name, param in signature.parameters.items():
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            ret.append(FunctionArgument(name=name, required=param.default is inspect._empty))
    return ret


class DataUpdater(NamedTuple):
    function: Callable
    arguments: list[FunctionArgument]
    returned: list[str]

    def get_dependencies(self) -> list[Dependency]:
        ret: list[Dependency] = []
        for depended in self.arguments:
            for depender in self.returned:
                ret.append(Dependency(depender=depender, depended=depended.name))
        return ret


class NodeData:
    def __init__(self, dirty: bool, data: Any, updated_by: str | None):
        self.dirty = dirty
        self.data = data
        self.updated_by = updated_by

    @staticmethod
    def default() -> 'NodeData':
        return NodeData(True, None, None)


class DataManager:
    def __init__(self):
        self._data_graph = TwoWayDependencyGraph[NodeData]()
        self._updaters: dict[str, DataUpdater] = dict()

    def _get_args(self, args: list[FunctionArgument]) -> dict[str, Any] | Error:
        ret = {}
        for arg in args:
            value = self.get_data(arg.name)
            if isinstance(value, Error):
                if arg.required:
                    return Error(f"Error getting argument {arg.name}: '{value.description}'")
                continue
            if value is None:
                if arg.required:
                    return Error(f"No data stored for argument '{arg.name}'.")
                continue
            ret[arg.name] = value
        return ret

    def _run_updater(self, name: str) -> None | Error:
        if name not in self._updaters:
            return Error(f"Updater {name} doesn't exist.")
        updater = self._updaters[name]
        args = self._get_args(updater.arguments)
        if isinstance(args, Error):
            return Error(f"Failed to run updater '{name}': {args.description}")
        res = updater.function(**args)
        if isinstance(res, Error):
            return Error(f"Error running updater function '{name}': {res.description}")
        if not isinstance(res, dict):
            return Error(f"Expected data updater function '{name}' to return a dictionary.")
        for variable_updated in updater.returned:
            if variable_updated not in res:
                return Error(
                    f"Variable '{variable_updated}' not returned by updater function '{name}' which promised it.")
            err = self.set_data(variable_updated, res.pop(variable_updated))
            if isinstance(err, Error):
                return Error(f"Error while running updater '{name}': {err.description}")
        if len(res) > 0:
            variable_names = "', '".join(list(res.keys()))
            return Error(f"Updater function '{name}' returned unexpected variables: '{variable_names}'")
        return None

    def add_updater(self, name: str, function: Callable) -> None | Error:
        self._updaters[name] = DataUpdater(function=function, arguments=get_function_arguments(function),
                                           returned=function.returned)
        self._data_graph.add_dependencies(self._updaters[name].get_dependencies())
        for variable_updated in variables_updated:
            node = self._data_graph.get_node(variable_updated)
            if node.data is None:
                node.data = NodeData.default()
            if node.data.updated_by is None:
                node.data.updated_by = name
            else:
                return Error(
                    f"Variable {variable_updated} is already updated by {node.data.updated_by}; tried to add updater {name} which wants to update the same variable.")
        return None

    def get_data(self, name: str) -> Any | Error:
        node = self._data_graph.get_node(name)
        if node is None:
            return Error(f"Variable {node} doesn't exist.")
        if node.data is None:
            return Error(f"Node {node} has no data.")
        if node.data.dirty:
            if node.data.updated_by is None:
                return Error(f"Could not update dirty variable {name} as no updater exists.")
            err = self._run_updater(node.data.updated_by)
            if isinstance(err, Error):
                return Error(f"Error while getting data '{name}': {err.description}")
        return node.data.data

    def set_data(self, name: str, value: Any) -> None | Error:
        node = self._data_graph.get_node(name)
        if node is None:
            return Error(f"Variable {node} doesn't exist.")
        if node.data is None:
            node.data = NodeData.default()
        node.data.data = value
        node.data.dirty = False
        return None


def try_updater(fixed_image: float, moving_image: float) -> dict[str, Any]:
    return {"similarity": fixed_image * moving_image}


def main():
    data_manager = DataManager()

    err = data_manager.add_updater("similarity_metric", try_updater, ["similarity"])
    if isinstance(err, Error):
        print(err)
        return

    data_manager.set_data("fixed_image", 2.0)
    data_manager.set_data("moving_image", 3.0)

    print(data_manager.get_data("similarity"))


if __name__ == "__main__":
    main()
