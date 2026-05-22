import pathlib

import nibabel
import torch

from reg23_experiments.data.structs import Error
from ._data import OneSeries, SeriesDescription, Volume
from ._loader_base import VolumeLoader

__all__ = ["NiiVolumeLoader"]


class NiiVolumeLoader(VolumeLoader):
    @staticmethod
    def name() -> str:
        return ".nii"

    @staticmethod
    def series_available(path: pathlib.Path) -> dict[str, SeriesDescription] | OneSeries:
        if not path.is_file():
            return {}
        if path.suffix != ".nii":
            return {}
        return OneSeries(file_path=NiiVolumeLoader.name())

    @staticmethod
    def load(path: pathlib.Path, series: str | None) -> Volume | Error:
        if series is not None:
            return Error(f".nii files cannot contain multiple series.")
        img = nibabel.load(str(path))
        if not isinstance(img, nibabel.Nifti1Image):
            return Error(f"Unsupported filetype for {str(path)}: {img.__class__.__name__}")
        data = torch.tensor(img.get_data())
        data = data.permute(*reversed(range(data.ndim)))
        spacing = torch.tensor(img.header.get_zooms(), dtype=torch.float64)[0:3]
        return Volume(raw_data=data, spacing=spacing, uid=str(path))
