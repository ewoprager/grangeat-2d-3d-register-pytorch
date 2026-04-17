import importlib.util
import logging
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

logger = logging.getLogger(__name__)


def get_data_config() -> data_utils.DataConfig:
    return data_utils.DataConfig(ARCHIVE_ROOT / "data_description.json")


def load_ct() -> tuple[torch.Tensor, torch.Tensor] | Error:
    ct_path = ARCHIVE_ROOT / get_data_config().get_ct_path()
    if not ct_path.is_file():
        return Error(f"CT file '{str(ct_path)}' not found.")
    image = sitk.ReadImage(str(ct_path))
    spacing = torch.tensor(image.GetSpacing())
    data = torch.tensor(sitk.GetArrayFromImage(image), dtype=torch.float32)
    data = data.flip(dims=(0,))
    return data, spacing


def load_xray(view_id, ct_volume_size: torch.Size, ct_volume_spacing: torch.Tensor) -> dict[str, Any] | Error:
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
    # data = data.flip(dims=(2,))
    pose_path = ARCHIVE_ROOT / get_data_config().get_optimized_pose_path(view_id)
    euler_3d_transform = sitk.ReadTransform(pose_path)
    source_distance: float = 968.1612

    translation = torch.tensor(euler_3d_transform.GetTranslation())
    translation[2] -= source_distance
    # translation *= -1.0
    rotation_matrix = torch.tensor(euler_3d_transform.GetMatrix()).view(3, 3)

    p = torch.tensor([  #
        [1.0, 0.0, 0.0],  #
        [0.0, 1.0, 0.0],  #
        [0.0, 0.0, -1.0]  #
    ])
    offset_2d = 0.5 * spacing[0:2] * torch.tensor([data.size()[1], -data.size()[0]], dtype=torch.float64)
    offset_3d = 0.5 * torch.tensor(ct_volume_size, dtype=translation.dtype).flip(dims=(0,)) * ct_volume_spacing.to(
        device=translation.device)
    offset_3d[2] *= -1.0

    import itertools
    # params:
    dim_sd_shift: int = 0
    sign_sd_shift: float = 1.0
    do_permute_t: bool = True
    sign_offset_2d: float = 1.0
    sign_offset_3d: float = 1.0
    do_offset_2d: bool = True

    def shift_by_source_distance():
        nonlocal translation, dim_sd_shift, sign_sd_shift, source_distance
        translation[dim_sd_shift] += sign_sd_shift * source_distance

    def permute_rot_mat():
        nonlocal rotation_matrix, p
        rotation_matrix = p @ rotation_matrix @ p

    def permute_t():
        nonlocal do_permute_t, translation, p
        if do_permute_t:
            translation = p @ translation

    def shift_offset_2d():
        nonlocal translation, offset_2d, sign_offset_2d, do_offset_2d
        if do_offset_2d:
            translation[0:2] += sign_offset_2d * offset_2d

    def shift_offset_3d():
        nonlocal translation, offset_3d, sign_offset_3d
        translation += sign_offset_3d * offset_3d

    def invert_t():
        nonlocal translation, rotation_matrix
        translation = -(rotation_matrix @ translation)

    func_perms = itertools.permutations(
        [shift_by_source_distance, permute_rot_mat, permute_t, shift_offset_2d, shift_offset_3d, invert_t])
    for (_dim_sd_shift, _sign_sd_shift, _do_permute_t, _sign_offset_2d, _sign_offset_3d, _do_offset_2d,
         funcs) in itertools.product(
            #
            [0, 1, 2],  #
            [-1.0, 1.0],  #
            [False, True],  #
            [-1.0, 1.0],  #
            [-1.0, 1.0],  #
            [False, True],  #
            func_perms  #
    ):
        dim_sd_shift = _dim_sd_shift
        sign_sd_shift = _sign_sd_shift
        do_permute_t = _do_permute_t
        sign_offset_2d = _sign_offset_2d
        sign_offset_3d = _sign_offset_3d
        do_offset_2d = _do_offset_2d
        for func in funcs:
            func()

    rotation = kornia.geometry.rotation_matrix_to_axis_angle(rotation_matrix)

    fixed_image_offset = spacing * (0.5 * torch.tensor(data.size(), dtype=torch.float32).flip(dims=(0,)) - torch.tensor(
        [471.22624, 492.11536]))
    logger.info(f"fixed image offset is {fixed_image_offset}")

    return {  #
        "image": data,  #
        "spacing": spacing,  #
        "scene_geometry": SceneGeometry(  #
            source_distance=source_distance,  #
            fixed_image_offset=fixed_image_offset  #
        ),  #
        "transformation": Transformation(rotation=rotation, translation=translation)  #
    }
