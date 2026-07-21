import pathlib

import nrrd
import torch

from reg23_experiments.data.structs import Error

from ._data import OneSeries, SeriesDescription, Volume
from ._loader_base import VolumeLoader

__all__ = ["NrrdVolumeLoader"]


class NrrdVolumeLoader(VolumeLoader):
    @staticmethod
    def name() -> str:
        return ".nrrd"

    @staticmethod
    def series_available(path: pathlib.Path) -> dict[str, SeriesDescription] | OneSeries:
        if not path.is_file():
            return {}
        if path.suffix != ".nrrd":
            return {}
        return OneSeries(file_type=NrrdVolumeLoader.name())

    @staticmethod
    def load(path: pathlib.Path, series: str | None) -> Volume | Error:
        if series is not None:
            return Error(f".nrrd files cannot contain multiple series.")
        data, header = nrrd.read(str(path))
        data = torch.tensor(data)
        if len(data.size()) > 3:
            data = data.squeeze()
        if len(data.size()) != 3:
            return Error(f"Expected CT volume data to be 3 dimensional; found image of size {data.size()}.")
        directions = torch.tensor(header['space directions'], dtype=torch.float64)
        spacing = directions.norm(dim=1).flip(dims=(0,))
        if "space origin" in header:
            image_position_patient = torch.tensor(header['space origin'], dtype=torch.float64)
        else:
            image_position_patient = None
        return Volume(raw_data=data, spacing=spacing, uid=str(path), image_position_patient=image_position_patient)
