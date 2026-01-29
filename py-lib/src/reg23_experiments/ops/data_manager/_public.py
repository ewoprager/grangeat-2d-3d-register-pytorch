import functools
import logging
from typing import Any, Callable

from reg23_experiments.data.structs import Error
from reg23_experiments.utils.reflection import FunctionArgument

from ._core import Updater, DAG, DataManagerSingleton
from ._helpers import takes_positional_args

__all__ = ["dag_updater", "args_from_dag", "init_data_manager", "data_manager"]

logger = logging.getLogger(__name__)


def dag_updater(*, names_returned: list[str]) -> Callable[[Callable], Updater]:
    """
    A decorator for turning a function into an `Updater` for use in a `DAG`. The named arguments of the function will be
    interpreted as nodes from which to read data from the `DAG`. The return value of the function must be a dictionary
    mapping string 'names' to values, where the strings must match the contents of the list `names_returned`. The
    returned names will be used as node names for updating values in the `DAG` with the returned values.
    :param names_returned: A list of the names of the values returned by the function.
    :return: An instance of `Updater`, built using the decorated function.
    """

    def decorator(function) -> Updater:
        return Updater.build(function=function, returned=names_returned)

    return decorator


def args_from_dag(*, names_left: list[str] | None = None):
    """
    A decorator for indicating that a function's arguments should be read from the `DAG`. The named arguments of the
    function will be interpreted as nodes from which to read data from the `DAG`. The return value will not be
    modified/intercepted, but an `Error` may be returned instead in the case of a failure to get any argument from
    the `DAG`. Arguments that should be left alone can be listed in the decorator argument `names_left`.
    :param names_left: A list of the arguments to leave in the function's signature (i.e. to not get from the DAG)

    Example use with no `names_left`:
    ```
    @args_from_dag()
    def subtract_moving_from_fixed(moving_image: torch.Tensor, fixed_image: torch.Tensor) -> torch.Tensor:
        return fixed_image - moving_image
    ```
    will be effectively be turned into:
    ```
    from program import data_manager
    def subtract_moving_from_fixed() -> torch.Tensor:
        moving_image = data_manager().get("moving_image")
        fixed_image = data_manager().get("fixed_image")
        return fixed_image - moving_image
    ```

    Example use with some `names_left`:
    ```
    @args_from_dag(names_left = ["moving_image"])
    def subtract_moving_from_fixed(moving_image: torch.Tensor, fixed_image: torch.Tensor) -> torch.Tensor:
        return fixed_image - moving_image
    ```
    will be effectively be turned into:
    ```
    from program import data_manager
    def subtract_moving_from_fixed(moving_image: torch.Tensor) -> torch.Tensor:
        fixed_image = data_manager().get("fixed_image")
        return fixed_image - moving_image
    ```
    """
    if names_left is None:
        names_left = []

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
            for name in names_left:
                if name not in kwargs:
                    return Error(f"Argument '{name}' specified in `names_left` not provided to function.")
            # get appropriate args from the DAG
            arguments_to_get = [argument for argument in arguments if argument.name not in names_left]
            from_dag = data_manager().get_args(arguments_to_get)
            if isinstance(from_dag, Error):
                return Error(f"Failed to get arguments to run function from dag: {from_dag.description}")
            # execute the function
            return function(**from_dag, **kwargs)

        return wrapper

    return decorator


def init_data_manager(**kwargs) -> DAG:
    """
    If the data manager singleton has yet to be initialised, initialises it with the given keyword arguments. If the
    data manager singleton has already been initialised, behaves exactly like `data_manager()`.
    :param kwargs: Keyword arguments to pass to the constructor of the singleton `DAG` instance.
    :return: The singleton `DAG` instance.
    """
    return DataManagerSingleton.instance().get(**kwargs)


def data_manager() -> DAG:
    """
    If the data manager singleton has yet to be initialised, initialises it with no arguments.
    :return: The singleton `DAG` instance.
    """
    return DataManagerSingleton.instance().get()


@dag_updater(names_returned=["similarity"])
def try_updater(fixed_image: float, moving_image: float) -> dict[str, Any]:
    return {"similarity": fixed_image * moving_image}


def main():
    init_data_manager()

    err = data_manager().add_updater("similarity_metric", try_updater)
    if isinstance(err, Error):
        print(err)
        return

    data_manager().set_data("fixed_image", 2.0)
    data_manager().set_data("moving_image", 3.0)

    # print(f"Data manager:\n{data_manager()}")

    print(data_manager().get("similarity"))


if __name__ == "__main__":
    main()
