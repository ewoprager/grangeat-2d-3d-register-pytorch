import pathlib

import nrrd
import torch

from reg23_experiments.data.structs import Error
from ._loader_base import Volume, VolumeLoader

__all__ = ["NrrdVolumeLoader"]


class NrrdVolumeLoader(VolumeLoader):
    def series_available(self, path: pathlib.Path) -> list[int]:
        if not path.is_file():
            return []
        if path.suffix != ".nrrd":
            return []
        return [0]

    def load(self, path: pathlib.Path, series: int) -> Volume | Error:
        data, header = nrrd.read(str(path))
        data = torch.tensor(data)
        directions = torch.tensor(header['space directions'], dtype=torch.float64)
        spacing = directions.norm(dim=1).flip(dims=(0,))
        return Volume(raw_data=data, spacing=spacing)
