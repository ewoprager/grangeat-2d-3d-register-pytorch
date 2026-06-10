from typing import Callable

from reg23_experiments.data.structs import Error

__all__ = ["get_string_required"]


def get_string_required(prompt: str, predicate: Callable[[str], None | Error]) -> str:
    prefix = ""
    while True:
        ret = input(prefix + prompt)
        err = predicate(ret)
        if isinstance(err, Error):
            prefix = err.description + ";\n"
        else:
            return ret
