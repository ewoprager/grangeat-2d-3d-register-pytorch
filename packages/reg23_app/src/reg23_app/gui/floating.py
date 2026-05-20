from typing import Callable

from magicgui.widgets import request_values

from reg23_experiments.data.structs import Error

__all__ = ["get_string_required"]


def get_string_required(*, message: str, validator: Callable[[str], Error | None] | None = None) -> str:
    message_prefix: str | None = None
    while True:
        prompt = message
        if message_prefix is not None:
            prompt = message_prefix + ";\n" + prompt
        values = request_values(value={"annotation": str, "label": prompt})
        if not values or not (value := values["value"]):
            message_prefix = "A value is required."
            continue
        if validator is not None and isinstance(err := validator(value), Error):
            message_prefix = err.description
            continue
        break
    return value
