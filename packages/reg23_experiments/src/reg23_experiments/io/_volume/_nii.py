import pathlib

import nibabel
import torch

from reg23_experiments.data.structs import Error
from ._loader_base import Volume, VolumeLoader

__all__ = ["NiiVolumeLoader"]


class NiiVolumeLoader(VolumeLoader):
    def series_available(self, path: pathlib.Path) -> list[int]:
        if not path.is_file():
            return []
        if path.suffix != ".nii":
            return []
        return [0]

    def load(self, path: pathlib.Path, series: int) -> Volume | Error:
        img = nibabel.load(str(path))
        data = torch.tensor(img.get_fdata())
        data = data.permute(*reversed(range(data.ndim)))
        spacing = torch.tensor(img.header.get_zooms(), dtype=torch.float64)[0:3]
        return Volume(raw_data=data, spacing=spacing)
