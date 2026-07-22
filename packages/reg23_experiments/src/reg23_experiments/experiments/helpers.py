import inspect
import pathlib
import pprint
import types
from datetime import datetime
from typing import Any, Callable, Literal

import torch

from reg23_experiments.ops import similarity_metric

__all__ = ["configs_to_dict", "save_dict", "instance_output_directory", "ParametrisedSimilarityMetric",
           "string_to_sim_met"]


def configs_to_dict(*vargs) -> dict[str, Any]:
    # convert all function pointers to their `str` names and merge all configs
    return {k: (v.__qualname__ if isinstance(v, types.FunctionType) else v) for config in vargs for k, v in
            config.trait_values().items()}


def save_dict(d: dict, *, directory: pathlib.Path, stem: str) -> None:
    directory.mkdir(exist_ok=True, parents=True)
    torch.save(d, directory / f"{stem}.pkl")
    (directory / f"{stem}.txt").write_text(pprint.pformat(d))


def instance_output_directory(script_output_directory: str | pathlib.Path) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ret: pathlib.Path = pathlib.Path(script_output_directory) / timestamp
    ret.mkdir(parents=True, exist_ok=True)
    return ret


class ParametrisedSimilarityMetric:
    def __init__(self, underlying_function: Callable, **kwargs):
        # filter out key-word arguments that the function doesn't accept
        self._underlying_function = underlying_function
        sig = inspect.signature(self._underlying_function)
        self._kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    @property
    def func(self) -> Callable:
        return lambda *args, **kwargs: self._underlying_function(*args, **kwargs, **self._kwargs)

    @property
    def func_weighted(self) -> Callable | None:
        if self._underlying_function == similarity_metric.ncc:
            weighted_function = similarity_metric.weighted_ncc
        elif self._underlying_function == similarity_metric.local_ncc:
            weighted_function = similarity_metric.weighted_local_ncc
        else:
            return None
        return lambda *args, **kwargs: weighted_function(*args, **kwargs, **self._kwargs)


def string_to_sim_met(  #
        config_string: str,  #
        *,  #
        kernel_size: int = 8,  #
        llambda: float = 1.0,  #
        gradient_method: Literal["sobel", "central_difference"] = "sobel"  #
) -> ParametrisedSimilarityMetric:
    if config_string == "zncc":
        return ParametrisedSimilarityMetric(similarity_metric.ncc)
    elif config_string == "local_zncc":
        return ParametrisedSimilarityMetric(similarity_metric.local_ncc, kernel_size=kernel_size)
    elif config_string == "multiscale_zncc":
        return ParametrisedSimilarityMetric(similarity_metric.multiscale_ncc, kernel_size=kernel_size, llambda=llambda)
    elif config_string == "gradient_correlation":
        return ParametrisedSimilarityMetric(similarity_metric.gradient_correlation, gradient_method=gradient_method)
    raise ValueError(f"Unknown similarity metric '{config_string}'.")
