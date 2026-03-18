import importlib.util
from typing import Any

import SimpleITK as sitk
import kornia.geometry
import numpy as np
import torch
import pathlib

from reg23_experiments.data.structs import Error, SceneGeometry, Transformation

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
    data = data.flip(dims=(1,))
    return data, spacing


def load_xray(view_id) -> dict[str, Any] | Error:
    """

    :param view_id:
    :return: dict containing "image": torch.Tensor, "spacing": torch.Tensor, "scene_geometry": SceneGeometry,
    "transformation": Transformation
    """
    xray_path = ARCHIVE_ROOT / get_data_config().get_mhd_images_path(view_id)[
        -1]  # ToDo: Note this takes the highest energy X-ray image; multiple are actually available for each
    if not xray_path.is_file():
        return Error(f"X-ray image file '{str(xray_path)}' not found.")
    image = sitk.ReadImage(str(xray_path))
    spacing = torch.tensor([0.2904, 0.2904])
    data = torch.tensor(sitk.GetArrayFromImage(image), dtype=torch.float32)[0]
    # data = data.flip(dims=(1,))
    pose_path = ARCHIVE_ROOT / get_data_config().get_optimized_pose_path(view_id)
    euler_3d_transform = sitk.ReadTransform(pose_path).GetInverse()
    source_distance: float = 968.1612
    euler_3d_transform.SetCenter([0.0, 0.0, -source_distance]) # ToDo: Is this the key?
    translation = torch.tensor(euler_3d_transform.GetTranslation())
    translation[0] -= 0.5 * spacing[0] * torch.tensor(data.size()[0], dtype=torch.float64)
    translation[2] -= 0.5 * spacing[1] * torch.tensor(data.size()[1], dtype=torch.float64)
    rotation_matrix = torch.tensor(euler_3d_transform.GetMatrix()).view(3, 3)

    p = torch.tensor([  #
        [-1.0, 0.0, 0.0],  #
        [0.0, 0.0, -1.0],  #
        [0.0, -1.0, 0.0]  #
    ])
    translation = p @ translation
    rotation_matrix = p @ rotation_matrix @ p.T

    translation[2] += source_distance

    rotation = kornia.geometry.rotation_matrix_to_axis_angle(rotation_matrix)

    return {  #
        "image": data,  #
        "spacing": spacing,  #
        "scene_geometry": SceneGeometry(  #
            source_distance=source_distance,  #
            fixed_image_offset=spacing * (500.0 - torch.tensor([471.22624, 492.11536]))  #
        ),  #
        "transformation": Transformation(rotation=rotation, translation=translation)  #
    }
