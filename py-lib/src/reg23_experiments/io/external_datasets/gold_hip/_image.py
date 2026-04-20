import importlib.util
import logging
from typing import Any
import traitlets
import math
from pprint import pprint

import SimpleITK as sitk
import kornia.geometry
import numpy as np
import torch
import pathlib

from pandas.core.window import doc
from tqdm import tqdm

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


class TransConvParams(traitlets.HasTraits):
    dim_sd_shift: int = traitlets.Integer()
    sign_sd_shift: float = traitlets.Float()
    do_permute_t: bool = traitlets.Bool()
    sign_offset_2d: float = traitlets.Float()
    sign_offset_3d: float = traitlets.Float()
    flip_offset_3d2: bool = traitlets.Bool()
    do_permute_rm: bool = traitlets.Bool()
    sign_sd_shift2: float = traitlets.Float()

    def __str__(self) -> str:
        return ("TransConvParams(dim_sd_shift={}, sign_sd_shift={:.1f}, do_permute_t={}, sign_offset_2d={:.1f}, "
                "sign_offset_3d={:.1f}, flip_offset_3d2={}, do_permute_rm={}, sign_sd_shift2={})").format(  #
            self.dim_sd_shift, self.sign_sd_shift, self.do_permute_t, self.sign_offset_2d, self.sign_offset_3d,
            self.flip_offset_3d2, self.do_permute_rm, self.sign_sd_shift2)


class TransConvValues(traitlets.HasTraits):
    rotation_matrix: torch.Tensor = traitlets.Instance(torch.Tensor)
    p: torch.Tensor = traitlets.Instance(torch.Tensor)
    translation: torch.Tensor = traitlets.Instance(torch.Tensor)
    source_distance: float = traitlets.Float()
    offset_2d: torch.Tensor = traitlets.Instance(torch.Tensor)
    offset_3d: torch.Tensor = traitlets.Instance(torch.Tensor)


def shift_by_source_distance(v: TransConvValues, p: TransConvParams) -> TransConvValues:
    v.translation[p.dim_sd_shift] += p.sign_sd_shift * v.source_distance
    return v


def shift_by_source_distance2(v: TransConvValues, p: TransConvParams) -> TransConvValues:
    v.translation[p.dim_sd_shift] += p.sign_sd_shift2 * v.source_distance
    return v


def permute_rot_mat(v: TransConvValues, p: TransConvParams) -> TransConvValues:
    if p.do_permute_rm:
        v.rotation_matrix = v.p @ v.rotation_matrix @ v.p
    return v


def permute_t(v: TransConvValues, p: TransConvParams) -> TransConvValues:
    if p.do_permute_t:
        v.translation = v.p @ v.translation
    return v


def shift_offset_2d(v: TransConvValues, p: TransConvParams) -> TransConvValues:
    v.translation[0:2] += p.sign_offset_2d * v.offset_2d
    return v


def shift_offset_3d(v: TransConvValues, p: TransConvParams) -> TransConvValues:
    if p.flip_offset_3d2:
        v.offset_3d[2] *= -1.0
    v.translation += p.sign_offset_3d * v.offset_3d
    return v


def invert_t(v: TransConvValues, p: TransConvParams) -> TransConvValues:
    v.translation = -(v.rotation_matrix @ v.translation)
    return v


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
    # offset_3d[2] *= -1.0

    if True:
        desired_t = torch.tensor(
            [76.36, -83.54, 178.00])  # Transformation(rotation=torch.tensor([-0.172, -2.091, -2.197]),
        # translation=torch.tensor([76.36, -83.54, 178.00]))

        min_dist = torch.inf
        min_params: list[tuple[float, TransConvParams, list[str]]] = []
        epsilon = 0.3

        import itertools
        func_list = [shift_by_source_distance, shift_by_source_distance2, permute_rot_mat, permute_t, shift_offset_2d,
                     shift_offset_3d, invert_t]
        func_perms = itertools.permutations(func_list)
        values = [[0, 1, 2],  #
                  [-1.0, 1.0],  #
                  [False, True],  #
                  [-1.0, 0.0, 1.0],  #
                  [-1.0, 0.0, 1.0],  #
                  [False, True],  #
                  [False, True],  #
                  [-1.0, 0.0, 1.0],  #
                  func_perms]
        total = math.prod([len(it) for it in values[:-1]]) * math.perm(len(func_list), len(func_list))
        for (dim_sd_shift, sign_sd_shift, do_permute_t, sign_offset_2d, sign_offset_3d, flip_offset_3d2, do_permute_rm,
             sign_sd_shift2, funcs) in tqdm(itertools.product(*values), total=total):
            params = TransConvParams(dim_sd_shift=dim_sd_shift, sign_sd_shift=sign_sd_shift, do_permute_t=do_permute_t,
                                     sign_offset_2d=sign_offset_2d, sign_offset_3d=sign_offset_3d,
                                     flip_offset_3d2=flip_offset_3d2, do_permute_rm=do_permute_rm,
                                     sign_sd_shift2=sign_sd_shift2)
            values = TransConvValues(rotation_matrix=rotation_matrix.clone(), p=p.clone(),
                                     translation=translation.clone(), source_distance=source_distance,
                                     offset_2d=offset_2d.clone(), offset_3d=offset_3d.clone())
            for func in funcs:
                values = func(values, params)

            # tr = values.translation
            # print("R=({:.2f}, {:.2f}, {:.2f}), T=({:.1f}, {:.1f}, {:.1f})".format(  #
            #     rt[0].item(), rt[1].item(), rt[2].item(), tr[0].item(), tr[1].item(), tr[2].item(),
            #
            # ))

            dist = torch.linalg.vector_norm(values.translation - desired_t)
            if dist < 5.0:
                func_names = [f.__name__ for f in funcs]
                min_dist = dist
                min_params = [(dist.item(), params, func_names)]
                break
            if dist < min_dist + epsilon:
                func_names = [f.__name__ for f in funcs]
                if dist < min_dist - epsilon:
                    min_params = [(dist.item(), params, func_names)]
                    min_dist = dist
                else:
                    min_params.append((dist.item(), params, func_names))

        print("Min dist = {:.3f}".format(min_dist))
        print("Minimum params:\n{}".format(pprint([(d, ps.trait_values(), l) for d, ps, l in min_params])))

        exit(0)

    # translation[2] -= source_distance

    translation[2] -= source_distance
    translation[0:2] += offset_2d
    rotation_matrix = p @ rotation_matrix @ p
    #
    translation = -(rotation_matrix @ translation)
    #
    offset_3d[2] *= -1.0
    translation -= offset_3d
    #
    # translation = p @ translation

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
