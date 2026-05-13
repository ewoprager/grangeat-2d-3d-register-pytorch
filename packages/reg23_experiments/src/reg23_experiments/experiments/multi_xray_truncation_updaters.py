import os
from typing import Any, Sequence

os.environ["QT_API"] = "PyQt6"

import torch
import pathlib
from jaxtyping import Float32, Float64

from reg23_experiments.io.volume import load_ct
from reg23_experiments.ops.data_manager import dadg_updater
from reg23_experiments.data.structs import Transformation, SceneGeometry
from reg23_experiments.ops import geometry
from reg23_experiments.ops.volume import downsample_trilinear_antialiased
from reg23_experiments.io.image import read_dicom

# ToDo: jax type the updaters?

__all__ = ["load_untruncated_ct", "set_target_image", "apply_truncation", "project_drr", "project_fiducials"]


@dadg_updater(names_returned=["untruncated_ct_volume", "ct_spacing", "ct_series_uid", "untruncated_ct_size"])
def load_untruncated_ct(*, ct_path: str, device: torch.device, ct_permutation: Sequence[int] | None = None) -> dict[
    str, Any]:
    ct_volume, ct_spacing, uid = load_ct(pathlib.Path(ct_path), check_for_dcm_suffix_if_dir=False)
    ct_volume = ct_volume.to(device=device, dtype=torch.float32)
    ct_spacing = ct_spacing.to(device=device)

    if ct_permutation is not None:
        assert len(ct_permutation) == 3
        ct_volume = ct_volume.permute(*ct_permutation)
        ct_spacing = ct_spacing[torch.tensor(ct_permutation)]

    return {"untruncated_ct_volume": ct_volume, "untruncated_ct_size": ct_volume.size(), "ct_spacing": ct_spacing,
            "ct_series_uid": uid}


@dadg_updater(names_returned=["source_distance", "image_2d_full", "image_2d_full_spacing", "transformation_gt",
                              "xray_sop_instance_uid"])
def set_target_image(*, xray_path: str, target_flipped: bool, device: torch.device) -> dict[str, Any]:
    # if xray_path is None:
    #     # generate a DRR through the volume
    #     drr_spec = None
    #     if not regenerate_drr:
    #         drr_spec = load_cached_drr(cache_directory, ct_path)
    #
    #     if drr_spec is None:
    #         tr = mapping_parameters_to_transformation(
    #             random_parameters_at_distance(mapping_transformation_to_parameters(ap_transformation),
    #                                           target_ap_distance))
    #         drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, untruncated_ct_volume, ct_spacing,
    #                                               save_to_cache=save_to_cache, size=new_drr_size, transformation=tr)
    #
    #     fixed_image_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
    #     del drr_spec
    #     uid = None
    # else:

    dicom = read_dicom(xray_path)
    image_2d_full, image_2d_full_spacing, scene_geometry, uid = (dicom["image"], dicom["spacing"],
                                                                 dicom["scene_geometry"], dicom["uid"])
    transformation_ground_truth = None

    if target_flipped:
        image_2d_full = image_2d_full.flip(dims=(1,))

    image_2d_full = image_2d_full.to(device=device)
    image_2d_full_spacing = image_2d_full_spacing.to(device=device)

    return {"source_distance": scene_geometry.source_distance, "image_2d_full": image_2d_full,
            "image_2d_full_spacing": image_2d_full_spacing, "transformation_gt": transformation_ground_truth,
            "xray_sop_instance_uid": uid}


@dadg_updater(names_returned=["ct_volumes"])
def apply_truncation(*, untruncated_ct_volume: Float32[torch.Tensor, "p q r"], truncation_percent: int) -> dict[
    str, Any]:
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
def project_drr(*, ct_volumes: list[torch.Tensor], ct_spacing: Float64[torch.Tensor, "3"],
                current_transformation: Transformation, fixed_image_size: torch.Size, source_distance: float,
                fixed_image_spacing: Float64[torch.Tensor, "2"], downsample_level: int,
                translation_offset: torch.Tensor, fixed_image_offset: Float64[torch.Tensor, "2"], device) -> dict[
    str, Any]:
    # Applying the translation offset
    new_translation = current_transformation.translation + torch.cat(
        (translation_offset.to(device=current_transformation.device),
         torch.tensor([0.0], device=device, dtype=current_transformation.translation.dtype)))
    transformation = Transformation(rotation=current_transformation.rotation, translation=new_translation).to(
        device=device)

    return {"moving_image": geometry.generate_drr(ct_volumes[downsample_level], transformation=transformation,
                                                  voxel_spacing=ct_spacing * 2.0 ** downsample_level,
                                                  detector_spacing=fixed_image_spacing,
                                                  scene_geometry=SceneGeometry(source_distance=source_distance,
                                                                               fixed_image_offset=fixed_image_offset),
                                                  output_size=fixed_image_size)}


@dadg_updater(names_returned=["projected_fiducials"])
def project_fiducials(*, current_transformation: Transformation, untruncated_ct_volume: Float32[torch.Tensor, "p q r"],
                      ct_spacing: Float64[torch.Tensor, "3"], fixed_image_offset: Float64[torch.Tensor, "2"],
                      translation_offset: Float64[torch.Tensor, "2"],
                      ct_fiducial_points: tuple[list[str], Float64[torch.Tensor, "3"]], source_distance: float) -> dict[
    str, Any]:
    # ToDo: Incorporate fixed image offset
    device = torch.device("cpu")
    current_transformation = current_transformation.to(device=device)
    # Applying the translation offset
    new_translation = current_transformation.translation + torch.cat(
        (translation_offset.to(device=device), torch.tensor([0.0], dtype=current_transformation.translation.dtype)))
    transformation = Transformation(rotation=current_transformation.rotation, translation=new_translation)
    input_points_3d = ct_fiducial_points[1].to(device=device) - 0.5 * ct_spacing.to(device=device) * torch.tensor(
        untruncated_ct_volume.size(), dtype=torch.float64).flip(dims=(0,))
    homo_vectors = torch.cat((input_points_3d, torch.ones((input_points_3d.size()[0], 1), dtype=torch.float64)), dim=1)
    transformed_points = torch.einsum("ji,ki->kj", transformation.get_h(device=device), homo_vectors)[:, 0:3]
    from_source = transformed_points - torch.tensor([[0.0, 0.0, -source_distance]], dtype=torch.float64)
    frac = torch.einsum("ji,i->j", from_source, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)) / source_distance
    projected = frac.unsqueeze(-1) * transformed_points[:, 0:2]
    return {"projected_fiducials": (ct_fiducial_points[0], projected)}
