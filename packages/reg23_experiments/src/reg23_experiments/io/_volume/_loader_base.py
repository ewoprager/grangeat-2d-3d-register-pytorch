import pathlib
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float64
from traitlets import traitlets

from reg23_experiments.data.structs import Error

__all__ = ["Volume", "VolumeLoader"]


class Volume(traitlets.HasTraits):
    raw_data: torch.Tensor = traitlets.Instance(torch.Tensor, allow_none=False)
    spacing: Float64[torch.Tensor, "3"] = traitlets.Instance(torch.Tensor, allow_none=False)


class VolumeLoader(ABC):
    @abstractmethod
    def series_available(self, path: pathlib.Path) -> list[int]:
        pass

    @abstractmethod
    def load(self, path: pathlib.Path, series: int) -> Volume | Error:
        pass
