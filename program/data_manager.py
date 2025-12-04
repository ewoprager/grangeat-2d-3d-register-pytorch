from typing import NamedTuple, Any, Callable, Generic, TypeVar
import inspect


class Error:
    def __init__(self, description: str):
        self.description = description


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


class NodeValue(Generic[T], NamedTuple):
    data: T | None
    depends_on: list[str]
    depended_on_by: list[str]

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


class DataUpdater:
    def __init__(self, function: Callable, variables_updated: list[str]):
        self._function = function
        signature = inspect.signature(function)
        self._variables_required: list[str] = [name for name, param in signature.parameters.items() if
                                               param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD,
                                                              param.KEYWORD_ONLY)]
        self._variables_updated = variables_updated

    def get_variables_updated(self) -> list[str]:
        return self._variables_updated

    def get_dependencies(self) -> list[Dependency]:
        ret: list[Dependency] = []
        for depended in self._variables_required:
            for depender in self._variables_updated:
                ret.append(Dependency(depender=depender, depended=depended))
        return ret

    def run(self, arg_getter: Callable[[str], Any], return_setter: Callable[[str, Any], None]) -> None | Error:
        returned = call_function_with_arg_getter(self._function, arg_getter)
        if not isinstance(returned, dict):
            return Error("Expected data updater function to return a dictionary.")
        for variable_updated in self._variables_updated:
            if variable_updated not in returned:
                return Error(f"Variable '{variable_updated}' not returned by updater function which promised it would.")
            return_setter(variable_updated, returned.pop(variable_updated))
        if len(returned) > 0:
            variable_names = "', '".join(list(returned.keys()))
            return Error(f"Updater function returned unexpected variables: '{variable_names}'")
        return None


class NodeData(NamedTuple):
    dirty: bool
    data: Any
    updated_by: str | None

    @staticmethod
    def default() -> 'NodeData':
        return NodeData(True, None, None)


class DataManager:
    def __init__(self):
        self._data_graph = TwoWayDependencyGraph[NodeData]()
        self._updaters: dict[str, DataUpdater] = dict()

    def add_updater(self, name: str, updater: DataUpdater) -> None | Error:
        self._updaters[name] = updater
        self._data_graph.add_dependencies(updater.get_dependencies())
        for variable_updated in updater.get_variables_updated():
            data = self._data_graph.get_node(variable_updated).data
            if data is not None:
                if data.updated_by is None:
                    data.updated_by = name
                else:
                    return Error(
                        f"Variable {variable_updated} is already updated by {data.updated_by}; tried to add updater {name} which wants to update the same variable.")
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
            self._updaters[node.data.updated_by].run(self.get_data, self.set_data)
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
