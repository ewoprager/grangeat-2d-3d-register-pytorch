from typing import NamedTuple, Callable
import inspect

__all__ = ["FunctionArgument"]


class FunctionArgument(NamedTuple):
    name: str
    required: bool
    annotation: type

    @staticmethod
    def get_for_function(function: Callable) -> list['FunctionArgument']:
        signature = inspect.signature(function)
        ret = []
        for name, param in signature.parameters.items():
            required = (param.default is inspect.Parameter.empty  # no default value
                        and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD))
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                ret.append(FunctionArgument(name=name, required=required, annotation=param.annotation))
        return ret
