import functools
import logging
from typing import Any, Callable

from reg23_experiments.data.structs import Error
from reg23_experiments.utils.reflection import FunctionArgument, takes_positional_args

from ._data import Updater
from ._dadg_standalone import StandaloneDADG, StandaloneDADGSingleton
from ._directed_acyclic_data_graph import DirectedAcyclicDataGraph

__all__ = ["dadg_updater", "args_from_dadg", "data_manager"]

logger = logging.getLogger(__name__)


def dadg_updater(*, names_returned: list[str]) -> Callable[[Callable], Updater]:
    """
    A decorator for turning a function into an `Updater` for use in an `IDirectedAcyclicDataGraph`. The named
    arguments of the function will be interpreted as nodes from which to read data from the
    `IDirectedAcyclicDataGraph`. The return value of the function must be a dictionary mapping string 'names' to
    values, where the strings must match the contents of the list `names_returned`. The returned names will be used
    as node names for updating values in the `IDirectedAcyclicDataGraph` with the returned values.
    :param names_returned: A list of the names of the values returned by the function.
    :return: An instance of `Updater`, built using the decorated function.
    """

    def decorator(function) -> Updater:
        return Updater.build(function=function, returned=names_returned)

    return decorator


def args_from_dadg(*, names_left: list[str] | None = None, dadg: DirectedAcyclicDataGraph | None = None,
                   namespace_captures: dict[str, str] | None = None):
    """
    A decorator for indicating that a function's arguments should be read from the `DAG`. The named arguments of the
    function will be interpreted as nodes from which to read data from the `DAG`. The return value will not be
    modified/intercepted, but an `Error` may be returned instead in the case of a failure to get any argument from
    the `DAG`. Arguments that should be left alone can be listed in the decorator argument `names_left`.
    :param names_left: [Optional] A list of the arguments to leave in the function's signature (i.e. to not get from the
    DAG)
    :param dadg: [Optional] Specify a DADG from which to pull the arguments. Will use the singleton DADG if unspecified
    :param namespace_captures: [Optional] A dict mapping variable names to the namespace they should be captured in. No
    namespace capture will be applied to any variable listed in `names_left`.

    Example use with no `names_left`:
    ```python
    @args_from_dag()
    def subtract_moving_from_fixed(moving_image: torch.Tensor, fixed_image: torch.Tensor) -> torch.Tensor:
        return fixed_image - moving_image
    ```
    will be effectively be turned into:
    ```python
    from program import data_manager
    def subtract_moving_from_fixed() -> torch.Tensor:
        moving_image = data_manager().get("moving_image")
        fixed_image = data_manager().get("fixed_image")
        return fixed_image - moving_image
    ```

    Example use with some `names_left` and some namespace captures:
    ```python
    @args_from_dag(names_left = ["moving_image"], namespace_captures = {"fixed_image": "ab"})
    def subtract_moving_from_fixed(moving_image: torch.Tensor, fixed_image: torch.Tensor) -> torch.Tensor:
        return fixed_image - moving_image
    ```
    will be effectively be turned into:
    ```python
    from program import data_manager
    def subtract_moving_from_fixed(moving_image: torch.Tensor) -> torch.Tensor:
        fixed_image = data_manager().get("ab__fixed_image")
        return fixed_image - moving_image
    ```
    """
    if names_left is None:
        names_left = []
    if namespace_captures is None:
        namespace_captures = dict({})

    def decorator(function):
        if takes_positional_args(function):
            raise ValueError(f"Functions decorated with `args_from_dag` must not take positional arguments. Prepend "
                             f"arguments with `*` to ensure this.")
        # all names specified in `named_left` must names of function arguments
        arguments = FunctionArgument.get_for_function(function)
        for name in names_left:
            if name not in [argument.name for argument in arguments]:
                raise KeyError(f"Every value in `names_left` must the name of one of the function's arguments. "
                               f"Unrecognised name '{name}'.")

        @functools.wraps(function)
        def wrapper(**kwargs) -> Any | Error:
            # check all left names are in **kwargs
            for _name in names_left:
                if _name not in kwargs:
                    return Error(f"Argument '{_name}' specified in `names_left` not provided to function.")
            # filter out arguments listed in `names_left`
            arguments_to_get = [argument for argument in arguments if argument.name not in names_left]
            # modify the names of arguments listed in `namespace_captures`
            for i in range(len(arguments_to_get)):
                if arguments_to_get[i].name in namespace_captures:
                    arguments_to_get[
                        i].name = f"{namespace_captures[arguments_to_get[i].name]}__{arguments_to_get[i].name}"
            # get the args from the DAG
            from_dag = (data_manager() if dadg is None else dadg).get_with_args(arguments_to_get)
            if isinstance(from_dag, Error):
                return Error(f"Failed to get arguments to run function from dag: {from_dag.description}")
            # modify the names of namespaced variables back
            for k, v in namespace_captures.items():
                namespaced_name = f"{v}__{k}"
                if namespaced_name in from_dag:
                    from_dag[k] = from_dag.pop(namespaced_name)
            # execute the function
            return function(**from_dag, **kwargs)

        return wrapper

    return decorator


def data_manager() -> StandaloneDADG:
    """
    If the data manager singleton has yet to be initialised, initialises it with no arguments.
    :return: The singleton `DAG` instance.
    """
    return StandaloneDADGSingleton.instance().get()


@dadg_updater(names_returned=["similarity"])
def try_updater(*, fixed_image: float, moving_image: float) -> dict[str, Any]:
    return {"similarity": fixed_image * moving_image}


def main():
    err = data_manager().add_updater("similarity_metric", try_updater)
    if isinstance(err, Error):
        print(err)
        return

    data_manager().set("fixed_image", 2.0)
    data_manager().set("moving_image", 3.0)

    # print(f"Data manager:\n{data_manager()}")

    print(data_manager().get("similarity"))


if __name__ == "__main__":
    main()
