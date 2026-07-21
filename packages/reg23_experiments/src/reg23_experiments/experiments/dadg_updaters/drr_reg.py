import logging
import os
from typing import Any, Sequence

os.environ["QT_API"] = "PyQt6"

import SimpleITK as sitk
import torch
from beartype import beartype as typechecker
from jaxtyping import Float32, Float64, jaxtyped

import reg23_core
from reg23_experiments.data.segmentation import NamedPoints2D, NamedPoints3D
from reg23_experiments.data.structs import Cropping, Error, SceneGeometry, Transformation
from reg23_experiments.io.image import XrayDICOM, load_cached_drr, read_dicom
from reg23_experiments.io.sitk import load_one_ct_series
from reg23_experiments.ops import ct, drr, geometry, volume
from reg23_experiments.ops.data_manager import dadg_updater
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters, random_parameters_at_distance

__all__ = ["load_untruncated_ct", "apply_truncation", "set_xray_target_image", "set_xray_target_image_with_no_gt",
           "set_synthetic_target_image", "refresh_image_2d_scale_factor", "refresh_hyperparameter_dependent",
           "refresh_mask_transformation_dependent", "project_drr", "project_fiducials"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["untruncated_ct_volume", "ct_spacing", "ct_series_uid"])
def load_untruncated_ct(  #
        *,  #
        ct_path: str,  #
        device: torch.device,  #
        ct_permutation: Sequence[int] | None = None  #
) -> dict[str, Any]:
    res: tuple[str, sitk.Image] | Error = load_one_ct_series(ct_path)
    if isinstance(res, Error):
        raise Exception(f"Failed to open CT from path '{ct_path}': {res.description}")
    uid, volume = res
    logger.info("CT loaded from path '{}' with size [{} x {} x {}] and spacing ({:.3f}, {:.3f}, {:.3f})".format(  #
        str(ct_path), *list(volume.GetSize()), *list(volume.GetSpacing())  #
    ))
    ct_volume = ct.convert_ct_to_mu_sitk(volume, dtype=torch.float32)
    if isinstance(ct_volume, Error):
        raise Exception(f"Failed to convert CT from path '{ct_path}' to mu: {ct_volume.description}")
    ct_volume = ct_volume.to(device=device)
    ct_spacing = torch.tensor(volume.GetSpacing(), device=device, dtype=torch.float64)

    if ct_permutation is not None:
        assert len(ct_permutation) == 3
        ct_volume = ct_volume.permute(*ct_permutation)
        ct_spacing = ct_spacing[torch.tensor(ct_permutation)]

    return {"untruncated_ct_volume": ct_volume, "ct_spacing": ct_spacing, "ct_series_uid": uid}


@dadg_updater(names_returned=["ct_volumes"])
def apply_truncation(  #
        *,  #
        untruncated_ct_volume: Float32[torch.Tensor, "p q r"],  #
        truncation_percent: int  #
) -> dict[str, Any]:
    # truncate the volume
    truncation_fraction = 0.01 * float(truncation_percent)
    top_bottom_chop = int(round(0.5 * truncation_fraction * float(untruncated_ct_volume.size()[0])))
    ct_volume = untruncated_ct_volume[
        top_bottom_chop:max(top_bottom_chop + 1, untruncated_ct_volume.size()[0] - top_bottom_chop)]
    # mipmap the volume
    ct_volumes = [ct_volume]
    level: int = 1
    while torch.tensor(ct_volumes[-1].size()).min() > 5:
        ct_volumes.append(volume.downsample_trilinear_antialiased(ct_volumes[0], scale_factor=0.5 ** float(level)))
        level += 1
    return {"ct_volumes": ct_volumes}


@dadg_updater(names_returned=["source_distance", "image_2d_full", "image_2d_full_spacing", "xray_sop_instance_uid"])
def set_xray_target_image(*, xray_path: str, device: torch.device) -> dict[str, Any]:
    dicom: XrayDICOM = read_dicom(xray_path)
    image_2d_full = dicom["image"].to(device=device, dtype=torch.float32)
    image_2d_full_spacing = dicom["spacing"].to(device=device, dtype=torch.float64)
    return {  #
        "source_distance": dicom["scene_geometry"].source_distance,  #
        "image_2d_full": image_2d_full,  #
        "image_2d_full_spacing": image_2d_full_spacing,  #
        "xray_sop_instance_uid": dicom["uid"]  #
    }


@dadg_updater(names_returned=["source_distance", "image_2d_full", "image_2d_full_spacing", "xray_sop_instance_uid",
                              "transformation_gt"])
def set_xray_target_image_with_no_gt(*, xray_path: str, device: torch.device) -> dict[str, Any]:
    dicom: XrayDICOM = read_dicom(xray_path)
    image_2d_full = dicom["image"].to(device=device, dtype=torch.float32)
    image_2d_full_spacing = dicom["spacing"].to(device=device, dtype=torch.float64)
    return {  #
        "source_distance": dicom["scene_geometry"].source_distance,  #
        "image_2d_full": image_2d_full,  #
        "image_2d_full_spacing": image_2d_full_spacing,  #
        "xray_sop_instance_uid": dicom["uid"],  #
        "transformation_gt": None,  #
    }


@dadg_updater(names_returned=["source_distance", "image_2d_full", "image_2d_full_spacing", "transformation_gt"])
def set_synthetic_target_image(  #
        *,  #
        ct_path: str,  #
        ct_spacing: torch.Tensor,  #
        untruncated_ct_volume: torch.Tensor,  #
        new_drr_size: torch.Size,  #
        regenerate_drr: bool,  #
        save_to_cache: bool,  #
        cache_directory: str,  #
        ap_transformation: Transformation,  #
        target_ap_distance: float  #
) -> dict[str, Any]:
    # generate a DRR through the volume
    drr_spec = None
    if not regenerate_drr:
        drr_spec = load_cached_drr(cache_directory, ct_path)

    if drr_spec is None:
        tr = mapping_parameters_to_transformation(
            random_parameters_at_distance(mapping_transformation_to_parameters(ap_transformation), target_ap_distance))
        drr_spec = drr.generate_drr_as_target(cache_directory, ct_path, untruncated_ct_volume, ct_spacing,
                                              save_to_cache=save_to_cache, size=new_drr_size, transformation=tr)

    image_2d_full_spacing, scene_geometry, image_2d_full, transformation_ground_truth = drr_spec
    del drr_spec

    return {"source_distance": scene_geometry.source_distance, "image_2d_full": image_2d_full,
            "image_2d_full_spacing": image_2d_full_spacing, "transformation_gt": transformation_ground_truth}


@dadg_updater(names_returned=["image_2d_scale_factor", "fixed_image_spacing"])
def refresh_image_2d_scale_factor(*, image_2d_full_spacing: Float64[torch.Tensor, "2"], downsample_level: int,
                                  ct_spacing: Float64[torch.Tensor, "3"]) -> dict[str, Any]:
    assert ct_spacing.device == image_2d_full_spacing.device
    downsampled_ct_spacing = ct_spacing * 2.0 ** float(downsample_level)
    image_2d_scale_factor = (image_2d_full_spacing.mean() / downsampled_ct_spacing.mean()).item()
    return {"image_2d_scale_factor": image_2d_scale_factor,
            "fixed_image_spacing": image_2d_full_spacing / image_2d_scale_factor}


@dadg_updater(names_returned=["cropped_target", "fixed_image_offset", "translation_offset", "fixed_image_size"])
@jaxtyped(typechecker=typechecker)
def refresh_hyperparameter_dependent(  #
        *,  #
        image_2d_full: Float32[torch.Tensor, "n m"],  #
        image_2d_full_spacing: Float64[torch.Tensor, "2"],  #
        cropping: Cropping | None,  #
        target_flipped: bool,  #
        source_offset: Float64[torch.Tensor, "2"],  #
        image_2d_scale_factor: float  #
) -> dict[str, Any]:
    device = image_2d_full.device
    assert source_offset.device == device
    assert image_2d_full_spacing.device == device

    # Flip the image 2d if necessary
    flipped_image_2d = image_2d_full.flip(dims=(1,)) if target_flipped else image_2d_full

    # Downsampling the image 2d
    scaled_image_2d = torch.nn.functional.interpolate(  #
        flipped_image_2d.unsqueeze(0).unsqueeze(0),  #
        scale_factor=image_2d_scale_factor,  #
        mode="bilinear",  #
        recompute_scale_factor=True,  #
        antialias=True  #
    )[0, 0]

    # Cropping for the fixed image
    if cropping is None:
        cropped_target = scaled_image_2d
        offset_from_cropping = torch.zeros(2, dtype=torch.float64, device=device)
    else:
        cropped_target = cropping.apply(scaled_image_2d)
        offset_from_cropping = (image_2d_full_spacing  #
                                * cropping.get_fractional_centre_offset(device=device)  #
                                * torch.tensor(image_2d_full.size(), dtype=torch.float64, device=device).flip(
                    dims=(0,)))

    # The fixed image is offset to adjust for the cropping, and according to the source offset
    # This isn't affected by downsample level
    fixed_image_offset = offset_from_cropping - source_offset

    # The translation offset prevents the source offset parameters from fighting the translation parameters in
    # the optimisation
    translation_offset = -source_offset

    return {"cropped_target": cropped_target, "fixed_image_offset": fixed_image_offset,
            "translation_offset": translation_offset, "fixed_image_size": cropped_target.size()}


@dadg_updater(names_returned=["mask", "fixed_image"])
def refresh_mask_transformation_dependent(*, ct_volumes: list[torch.Tensor], ct_spacing: Float64[torch.Tensor, "3"],
                                          cropped_target: Float32[torch.Tensor, "n m"],
                                          mask_transformation: Transformation | None,
                                          fixed_image_spacing: Float64[torch.Tensor, "2"],
                                          fixed_image_offset: Float64[torch.Tensor, "2"], source_distance: float,
                                          device) -> dict[str, Any]:
    device = ct_volumes[0].device
    assert ct_spacing.device == device
    assert cropped_target.device == device
    assert fixed_image_spacing.device == device
    assert fixed_image_offset.device == device

    if mask_transformation is None:
        mask = torch.ones_like(cropped_target)
        fixed_image = cropped_target
    else:
        assert mask_transformation.device == device
        mask = reg23_core.project_drr_cuboid_mask(  #
            volume_size=torch.tensor(ct_volumes[0].size(), device=device).flip(dims=(0,)),  #
            voxel_spacing=ct_spacing,  #
            homography_matrix_inverse=mask_transformation.inverse().get_h(device=device),  #
            source_distance=source_distance,  #
            output_width=cropped_target.size()[1],  #
            output_height=cropped_target.size()[0],  #
            output_offset=fixed_image_offset,  #
            detector_spacing=fixed_image_spacing  #
        )
        fixed_image = mask * cropped_target

    return {"mask": mask, "fixed_image": fixed_image}


@dadg_updater(names_returned=["moving_image"])
def project_drr(  #
        *,  #
        ct_volumes: list[torch.Tensor],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        current_transformation: Transformation,  #
        fixed_image_size: torch.Size,  #
        source_distance: float,  #
        fixed_image_spacing: Float64[torch.Tensor, "2"],  #
        downsample_level: int,  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"]  #
) -> dict[str, Any]:
    return {"moving_image": geometry.generate_drr(  #
        ct_volumes[downsample_level],  #
        transformation=current_transformation.with_translation_offset(translation_offset),  #
        voxel_spacing=ct_spacing * 2.0 ** downsample_level,  #
        detector_spacing=fixed_image_spacing,  #
        scene_geometry=SceneGeometry(source_distance=source_distance, fixed_image_offset=fixed_image_offset),  #
        output_size=fixed_image_size  #
    )}


@dadg_updater(names_returned=["projected_fiducials"])
def project_fiducials(  #
        *,  #
        current_transformation: Transformation,  #
        untruncated_ct_volume: Float32[torch.Tensor, "p q r"],  #
        ct_spacing: Float64[torch.Tensor, "3"],  #
        image_2d_full: Float32[torch.Tensor, "n m"],  #
        fixed_image_offset: Float64[torch.Tensor, "2"],  #
        translation_offset: Float64[torch.Tensor, "2"],  #
        image_2d_full_spacing: Float64[torch.Tensor, "2"],  #
        ct_fiducial_points: NamedPoints3D,  #
        source_distance: float,  #
) -> dict[str, Any]:
    device = torch.device("cpu")
    transformation = current_transformation.to(device=device).with_translation_offset(translation_offset)
    input_vectors = ct_fiducial_points.data.cpu() - 0.5 * ct_spacing.cpu() * torch.tensor(untruncated_ct_volume.size(),
                                                                                          dtype=torch.float64).flip(
        dims=(0,))
    projected = geometry.project_vectors(input_vectors, source_distance=source_distance, transformation=transformation)
    size_tensor = torch.tensor(image_2d_full.size(), dtype=torch.float64).flip(dims=(0,))
    output_points_2d = projected + 0.5 * image_2d_full_spacing.cpu() * size_tensor
    return {"projected_fiducials": NamedPoints2D(names=ct_fiducial_points.names, data=output_points_2d)}
