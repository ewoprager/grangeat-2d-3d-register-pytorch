import inspect
import pathlib
import pprint
import types
from datetime import datetime
from typing import Any, Callable, Literal

import torch

from reg23_experiments.ops import similarity_metric

__all__ = ["configs_to_dict", "save_dict", "instance_output_directory", "string_to_sim_met"]


def configs_to_dict(*vargs) -> dict[str, Any]:
    # convert all function pointers to their `str` names and merge all configs
    return {k: (v.__qualname__ if isinstance(v, types.FunctionType) else v) for config in vargs for k, v in
            config.trait_values().items()}


def save_dict(d: dict, *, directory: pathlib.Path, stem: str) -> None:
    directory.mkdir(exist_ok=True, parents=True)
    torch.save(d, directory / f"{stem}.pkl")
    (directory / f"{stem}.txt").write_text(pprint.pformat(d))


def instance_output_directory(script_output_directory: str | pathlib.Path, name: str | None = None) -> pathlib.Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = timestamp if name is None else f"{timestamp}_{name}"
    ret: pathlib.Path = pathlib.Path(script_output_directory) / dir_name
    ret.mkdir(parents=True, exist_ok=True)
    return ret


STRING_TO_SIM_MET = {  #
    "zncc": similarity_metric.ncc,  ##
    "gradient_correlation": similarity_metric.gradient_correlation,  #
    "mutual_information": similarity_metric.mutual_information,  #
}


def string_to_sim_met(  #
        name: str,  #
        *,  #
        kernel_size: int = 8,  #
        llambda: float = 1.0,  #
        gradient_method: Literal["sobel", "central_difference"] = "sobel",  #
        mi_bin_count: int = 64,  #
        dim: int | tuple | torch.Size | None = (-2, -1),  #
) -> Callable:
    all_kwargs = {  #
        "kernel_size": kernel_size,  #
        "llambda": llambda,  #
        "gradient_method": gradient_method,  #
        "mi_bin_count": mi_bin_count,  #
        "dim": dim,  #
    }
    if name not in STRING_TO_SIM_MET:
        raise ValueError(f"Unknown similarity metric '{name}'.")
    underlying = STRING_TO_SIM_MET[name]
    sig = inspect.signature(underlying)
    applicable_kwargs = {  #
        k: v  #
        for k, v in all_kwargs.items()  #
        if k in sig.parameters  #
    }
    return lambda *args, **kwargs: underlying(*args, **kwargs, **applicable_kwargs)
