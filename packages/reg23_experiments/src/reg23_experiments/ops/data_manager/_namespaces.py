import copy
import functools
from typing import Any, Callable

from reg23_experiments.utils.reflection import takes_positional_args
from ._data import Updater

__all__ = ["capture_in_namespaces"]


def capture_in_namespaces(namespace_captures: dict[str, str]) -> Callable[[Callable | Updater], Callable | Updater]:
    """
    A decorator that adds namespaces to all input and output variables for functions or Updaters to be used with a DADG.

    Capturing a variable in a namespace just means prepending its name with 'namespace__'.

    :param namespace_captures: The namespaces to introduce. This is a dict that maps a variable name to the namespace it
    should be captured in.

    Can be applied to functions, as long as they exclusively have keyword arguments, and a return type of `dict[str,
    Any]`.

    Can also be applied to instances of the `Updater` class, (which result from decorating functions with
    `dadg_updater`).

    Example use with a function:
    ```python
    @capture_in_namespaces({"image_1": "og", "image_2: "mean_subtracted"})
    def subtract_mean(*, image_1: torch.Tensor) -> dict[str, Any]:
        return {"image_2": image_1 - image_1.mean()}
    ```
    will be transformed into a function that behaves like this:
    ```python
    def subtract_mean(*, og__image_1: torch.Tensor) -> dict[str, Any]:
        return {"mean_subtracted__image_2": og__image_1 - og__image_1.mean()}
    ```
    """

    def decorator(function_or_updater: Callable | Updater) -> Callable | Updater:
        if isinstance(function_or_updater, Updater):
            # Decorating an updater, so return a modified updater
            updater = function_or_updater
            function = capture_in_namespaces(namespace_captures)(updater.function)
            returned = copy.deepcopy(updater.returned)
            arguments = copy.deepcopy(updater.arguments)
            dependencies = copy.deepcopy(updater.dependencies)
            for i in range(len(returned)):
                if returned[i] in namespace_captures:
                    returned[i] = f"{namespace_captures[returned[i]]}__{returned[i]}"
            for i in range(len(arguments)):
                if arguments[i].name in namespace_captures:
                    arguments[i].name = f"{namespace_captures[arguments[i].name]}__{arguments[i].name}"
            for i in range(len(dependencies)):
                if dependencies[i].depender in namespace_captures:
                    dependencies[
                        i].depender = f"{namespace_captures[dependencies[i].depender]}__{dependencies[i].depender}"
                if dependencies[i].depended in namespace_captures:
                    dependencies[
                        i].depended = f"{namespace_captures[dependencies[i].depended]}__{dependencies[i].depended}"
            return Updater(function=function, returned=returned, arguments=arguments, dependencies=dependencies)

        # Decorating a function, so return a modified function
        function = function_or_updater
        if takes_positional_args(function):
            raise ValueError(
                f"Functions decorated with `capture_in_namespaces` must not take positional arguments. Prepend "
                f"arguments with `*` to ensure this.")

        @functools.wraps(function)
        def wrapper(**kwargs) -> dict[str, Any]:
            # strip the namespacing from the keyword arguments according to the captures
            for key, value in namespace_captures.items():
                namespaced_key = f"{value}__{key}"
                if namespaced_key in kwargs:
                    kwargs[key] = kwargs.pop(namespaced_key)
            # call the function with the modified keyword arguments
            ret_dict = function(**kwargs)
            # introduce namespacing to the returned dict
            for key, value in namespace_captures.items():
                namespaced_key = f"{value}__{key}"
                if key in ret_dict:
                    ret_dict[namespaced_key] = ret_dict.pop(key)
            # return the modified dict
            return ret_dict

        return wrapper

    return decorator
