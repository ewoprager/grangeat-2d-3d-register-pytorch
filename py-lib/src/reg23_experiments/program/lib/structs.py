from typing import NamedTuple, Callable
import inspect
import traitlets

__all__ = ["Error", "FunctionArgument"]


class Error:
    def __init__(self, description: str):
        self.description = description

    def __str__(self) -> str:
        return f"{self.description}"

    def __repr__(self) -> str:
        return f"Error(description='{self.description}')"


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


class StrictHasTraits(traitlets.HasTraits):
    """
    A `HasTraits` subclass with strict `__init__` semantics:
      - unknown kwargs are rejected
      - traits with `default_value=Undefined` are required
    """

    def __init__(self, **kwargs):
        # reject unknown kwargs
        valid_traits = set(self.traits())
        unknown = set(kwargs) - valid_traits
        if unknown:
            cls = type(self).__name__
            raise TypeError(f"{cls} got unexpected keyword arguments: {unknown}")

        super().__init__(**kwargs)

        # enforce required traits
        missing = {  #
            name for name, trait in self.traits().items() if
            trait.default_value is traitlets.Undefined and getattr(self, name) is traitlets.Undefined  #
        }

        if missing:
            cls = type(self).__name__
            raise TypeError(f"{cls} missing required arguments: {missing}")
