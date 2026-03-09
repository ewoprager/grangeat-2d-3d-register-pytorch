import importlib.util

import SimpleITK as sitk
import numpy as np
import torch
import pathlib

from reg23_experiments.data.structs import Error, SceneGeometry

HERE = pathlib.Path(__file__).resolve().parent
ARCHIVE_ROOT = HERE / pathlib.Path("archive_download/researchdata/")

spec = importlib.util.spec_from_file_location("data_utils", ARCHIVE_ROOT / "code/data_utils.py")
data_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_utils)

__all__ = ["get_data_config", "load_ct", "load_xray"]


def get_data_config() -> data_utils.DataConfig:
    return data_utils.DataConfig(ARCHIVE_ROOT / "data_description.json")


def load_ct() -> tuple[torch.Tensor, torch.Tensor] | Error:
    ct_path = ARCHIVE_ROOT / get_data_config().get_ct_path()
    if not ct_path.is_file():
        return Error(f"CT file '{str(ct_path)}' not found.")
    image = sitk.ReadImage(str(ct_path))
    spacing = torch.tensor(image.GetSpacing())
    data = torch.tensor(sitk.GetArrayFromImage(image), dtype=torch.float32)
    return data, spacing


def load_xray(view_id) -> tuple[torch.Tensor, torch.Tensor, SceneGeometry] | Error:
    xray_path = ARCHIVE_ROOT / get_data_config().get_mhd_images_path(view_id)[
        -1]  # ToDo: Note this takes the highest energy X-ray image; multiple are actually available for each
    if not xray_path.is_file():
        return Error(f"X-ray image file '{str(xray_path)}' not found.")
    image = sitk.ReadImage(str(xray_path))
    spacing = torch.tensor([0.2904, 0.2904])
    data = torch.tensor(sitk.GetArrayFromImage(image), dtype=torch.float32)[0]
    return data, spacing, SceneGeometry(source_distance=968.1612,
                                        fixed_image_offset=spacing * (500.0 - torch.tensor([471.22624, 492.11536])))
