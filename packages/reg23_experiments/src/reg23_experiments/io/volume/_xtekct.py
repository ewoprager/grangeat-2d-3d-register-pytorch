import configparser
import pathlib

import tifffile
import torch
from tqdm import tqdm

from reg23_experiments.data.structs import Error

from ._data import OneSeries, SeriesDescription, Volume
from ._loader_base import VolumeLoader

__all__ = ["XTekCTVolumeLoader"]


class XTekCTVolumeLoader(VolumeLoader):
    CONFIG_FILE_SUFFIX: str = ".xtekVolume"
    CONFIG_SECTION: str = "XTekCT"

    @staticmethod
    def name() -> str:
        return "XTekCT"

    @staticmethod
    def series_available(path: pathlib.Path) -> dict[str, SeriesDescription] | OneSeries:
        if not path.is_dir():
            return {}
        if len([  #
            f for f in path.iterdir()  #
            if f.is_file() and f.suffix == XTekCTVolumeLoader.CONFIG_FILE_SUFFIX  #
        ]) != 1:
            return {}
        return OneSeries(file_type=XTekCTVolumeLoader.name())

    @staticmethod
    def load(path: pathlib.Path, series: str | None) -> Volume | Error:
        if series is not None:
            return Error(f"XTekCT directories cannot contain multiple series.")
        config_paths = [f for f in path.iterdir() if f.is_file() and f.suffix == XTekCTVolumeLoader.CONFIG_FILE_SUFFIX]
        if len(config_paths) != 1:
            return Error(f"Found {len(config_paths)} config files in CT directory '{path}'.")
        config = configparser.ConfigParser()
        config.read(config_paths[0])
        if XTekCTVolumeLoader.CONFIG_SECTION not in config:
            return Error(f"Expected section 'XTekCT' in config file '{config_paths[0]}', but not found.")
        if "voxelsizex" not in config[XTekCTVolumeLoader.CONFIG_SECTION]:
            return Error(f"Expected key 'voxelsizex' in section '{XTekCTVolumeLoader.CONFIG_SECTION}' in config file '"
                         f"{config_paths[0]}', but not found.")
        if "voxelsizey" not in config[XTekCTVolumeLoader.CONFIG_SECTION]:
            return Error(f"Expected key 'voxelsizey' in section '{XTekCTVolumeLoader.CONFIG_SECTION}' in config file '"
                         f"{config_paths[0]}', but not found.")
        if "voxelsizez" not in config[XTekCTVolumeLoader.CONFIG_SECTION]:
            return Error(f"Expected key 'voxelsizez' in section '{XTekCTVolumeLoader.CONFIG_SECTION}' in config file '"
                         f"{config_paths[0]}', but not found.")
        if "voxelsx" not in config[XTekCTVolumeLoader.CONFIG_SECTION]:
            return Error(f"Expected key 'voxelsx' in section '{XTekCTVolumeLoader.CONFIG_SECTION}' in config file '"
                         f"{config_paths[0]}', but not found.")
        if "voxelsy" not in config[XTekCTVolumeLoader.CONFIG_SECTION]:
            return Error(f"Expected key 'voxelsy' in section '{XTekCTVolumeLoader.CONFIG_SECTION}' in config file '"
                         f"{config_paths[0]}', but not found.")
        if "voxelsz" not in config[XTekCTVolumeLoader.CONFIG_SECTION]:
            return Error(f"Expected key 'voxelsz' in section '{XTekCTVolumeLoader.CONFIG_SECTION}' in config file '"
                         f"{config_paths[0]}', but not found.")
        volume = torch.empty((  #
            int(config[XTekCTVolumeLoader.CONFIG_SECTION]["voxelsz"]),  #
            int(config[XTekCTVolumeLoader.CONFIG_SECTION]["voxelsy"]),  #
            int(config[XTekCTVolumeLoader.CONFIG_SECTION]["voxelsx"])  #
        ), dtype=torch.int16)
        tif_paths = sorted([f for f in path.iterdir() if f.is_file() and (f.suffix == ".tif" or f.suffix == ".tiff")])
        if len(tif_paths) != volume.size()[0]:
            return Error(
                f"Config specified {volume.size()[0]} voxels in Z-direction, but found {len(tif_paths)} image files")
        for i, slice_path in tqdm(enumerate(tif_paths), desc="Reading .tif files", total=len(tif_paths)):
            volume[i] = torch.from_numpy(tifffile.imread(slice_path))
        spacing = torch.tensor([  #
            float(config[XTekCTVolumeLoader.CONFIG_SECTION]["voxelsizex"]),  #
            float(config[XTekCTVolumeLoader.CONFIG_SECTION]["voxelsizey"]),  #
            float(config[XTekCTVolumeLoader.CONFIG_SECTION]["voxelsizez"])  #
        ], dtype=torch.float64)
        return Volume(raw_data=volume, spacing=spacing, uid=str(path))
