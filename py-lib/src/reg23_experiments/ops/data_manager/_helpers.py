import inspect


def takes_positional_args(func) -> bool:
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.VAR_POSITIONAL,  # *args
                          ):
            return True
    return False
