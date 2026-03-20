from typing import Callable
import inspect
import traitlets

__all__ = ["FunctionArgument", "takes_positional_args"]


class FunctionArgument(traitlets.HasTraits):
    name: str = traitlets.Unicode(allow_none=False)
    required: bool = traitlets.Bool(allow_none=False)
    annotation: type = traitlets.Any()

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


def takes_positional_args(function: Callable) -> bool:
    sig = inspect.signature(function)
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.VAR_POSITIONAL,  # *args
                          ):
            return True
    return False
