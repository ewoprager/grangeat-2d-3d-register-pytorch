import pathlib
from abc import ABC, abstractmethod

from reg23_experiments.data.structs import Error
from ._data import OneSeries, SeriesDescription, Volume

__all__ = ["VolumeLoader"]


class VolumeLoader(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def series_available(path: pathlib.Path) -> dict[str, SeriesDescription] | OneSeries:
        pass

    @staticmethod
    @abstractmethod
    def load(path: pathlib.Path, series: str | None) -> Volume | Error:
        pass
