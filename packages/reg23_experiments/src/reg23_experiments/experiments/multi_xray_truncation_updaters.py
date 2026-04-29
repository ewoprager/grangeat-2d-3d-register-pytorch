import os
from typing import Any, Sequence

os.environ["QT_API"] = "PyQt6"

import torch
import pathlib
import pydicom

from reg23_experiments.io.volume import load_ct
from reg23_experiments.io.image import load_cached_drr
from reg23_experiments.ops.data_manager import dadg_updater
from reg23_experiments.ops.optimisation import mapping_transformation_to_parameters, \
    mapping_parameters_to_transformation, random_parameters_at_distance
from reg23_experiments.ops import drr
from reg23_experiments.data.structs import Transformation, SceneGeometry
from reg23_experiments.ops import geometry
from reg23_experiments.ops.volume import downsample_trilinear_antialiased
from reg23_experiments.io.image import read_dicom

__all__ = ["load_untruncated_ct", "set_target_image", "apply_truncation", "project_drr", "read_xray_uid"]


@dadg_updater(names_returned=["untruncated_ct_volume", "ct_spacing"])
def load_untruncated_ct(*, ct_path: str, device: torch.device, ct_permutation: Sequence[int] | None = None) -> dict[
    str, Any]:
    ct_volume, ct_spacing = load_ct(pathlib.Path(ct_path), check_for_dcm_suffix_if_dir=False)
    ct_volume = ct_volume.to(device=device, dtype=torch.float32)
    ct_spacing = ct_spacing.to(device=device)

    if ct_permutation is not None:
        assert len(ct_permutation) == 3
        ct_volume = ct_volume.permute(*ct_permutation)
        ct_spacing = ct_spacing[torch.tensor(ct_permutation)]

    return {"untruncated_ct_volume": ct_volume, "ct_spacing": ct_spacing}


@dadg_updater(names_returned=["source_distance", "image_2d_full", "fixed_image_spacing", "transformation_gt"])
def set_target_image(*, ct_path: str, ct_spacing: torch.Tensor, untruncated_ct_volume: torch.Tensor,
                     new_drr_size: torch.Size, regenerate_drr: bool, save_to_cache: bool, cache_directory: str,
                     ap_transformation: Transformation, target_ap_distance: float, xray_path: str | None,
                     target_flipped: bool, device: torch.device) -> dict[str, Any]:
    if xray_path is None:
        # generate a DRR through the volume
        drr_spec = None
        if not regenerate_drr:
            drr_spec = load_cached_drr(cache_directory, ct_path)

        if drr_spec is None:
            tr = mapping_parameters_to_transformation(
                random_parameters_at_distance(mapping_transformation_to_parameters(ap_transformation),
                                              target_ap_distance))
            drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, untruncated_ct_volume, ct_spacing,
                                                  save_to_cache=save_to_cache, size=new_drr_size, transformation=tr)

        fixed_image_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
        del drr_spec
    else:
        image_2d_full, fixed_image_spacing, scene_geometry = read_dicom(xray_path)
        transformation_ground_truth = None

    if target_flipped:
        image_2d_full = image_2d_full.flip(dims=(1,))

    image_2d_full = image_2d_full.to(device=device)

    return {"source_distance": scene_geometry.source_distance, "image_2d_full": image_2d_full,
            "fixed_image_spacing": fixed_image_spacing, "transformation_gt": transformation_ground_truth}


@dadg_updater(names_returned=["ct_volumes"])
def apply_truncation(*, untruncated_ct_volume: torch.Tensor, truncation_percent: int) -> dict[str, Any]:
    # truncate the volume
    truncation_fraction = 0.01 * float(truncation_percent)
    top_bottom_chop = int(round(0.5 * truncation_fraction * float(untruncated_ct_volume.size()[0])))
    ct_volume = untruncated_ct_volume[
        top_bottom_chop:max(top_bottom_chop + 1, untruncated_ct_volume.size()[0] - top_bottom_chop)]
    # mipmap the volume
    ct_volumes = [ct_volume]
    while torch.tensor(ct_volumes[-1].size()).min() > 3:
        ct_volumes.append(downsample_trilinear_antialiased(ct_volumes[-1], scale_factor=0.5))
    return {"ct_volumes": ct_volumes}


@dadg_updater(names_returned=["moving_image"])
def project_drr(*, ct_volumes: list[torch.Tensor], ct_spacing: torch.Tensor, current_transformation: Transformation,
                fixed_image_size: torch.Size, source_distance: float, fixed_image_spacing: torch.Tensor,
                downsample_level: int, translation_offset: torch.Tensor, fixed_image_offset: torch.Tensor,
                image_2d_scale_factor: float, device) -> dict[str, Any]:
    # Applying the translation offset
    new_translation = current_transformation.translation + torch.cat(
        (torch.tensor([0.0], device=device, dtype=current_transformation.translation.dtype),
         translation_offset.to(device=current_transformation.device)))
    transformation = Transformation(rotation=current_transformation.rotation, translation=new_translation).to(
        device=device)

    return {"moving_image": geometry.generate_drr(ct_volumes[downsample_level], transformation=transformation,
                                                  voxel_spacing=ct_spacing * 2.0 ** downsample_level,
                                                  detector_spacing=fixed_image_spacing / image_2d_scale_factor,
                                                  scene_geometry=SceneGeometry(source_distance=source_distance,
                                                                               fixed_image_offset=fixed_image_offset),
                                                  output_size=fixed_image_size)}


@dadg_updater(names_returned=["xray_sop_instance_uid"])
def read_xray_uid(*, xray_path: str | None) -> dict[str, Any]:
    if xray_path is None:
        uid = None
    else:
        uid = str(pydicom.dcmread(xray_path)["SOPInstanceUID"].value)
    return {"xray_sop_instance_uid": uid}
