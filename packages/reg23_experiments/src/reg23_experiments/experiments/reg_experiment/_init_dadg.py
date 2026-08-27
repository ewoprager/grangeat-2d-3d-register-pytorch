import logging
import pathlib
from collections.abc import Sequence

import SimpleITK as sitk
import torch

from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.experiments.experiment_set_config import Cartesian, Constant, ExperimentSetConfig
from reg23_experiments.io.sitk import load_ct_series
from reg23_experiments.ops.ct import convert_ct_to_mu_sitk
from reg23_experiments.ops.data_manager import DirectedAcyclicDataGraph, dadg_updater, data_manager

from ._dadg_updaters import batched
from ._dadg_updaters import drr_reg as updaters
from ._setup import ImageSpecificConfigurations

__all__ = ["init_dadg"]

logger = logging.getLogger(__name__)


def load_untruncated_ct(  #
        ct_path: pathlib.Path,  #
        ct_series_uid: str,  #
        device: torch.device,  #
        ct_permutation: Sequence[int] | None = None  #
) -> tuple[torch.Tensor, torch.Tensor] | Error:
    volume: sitk.Image | Error = load_ct_series(ct_path, ct_series_uid)
    if isinstance(volume, Error):
        return Error(f"Failed to open CT from path '{str(ct_path)}': {volume.description}")
    tensor: torch.Tensor | Error = convert_ct_to_mu_sitk(volume, dtype=torch.float32)
    if isinstance(tensor, Error):
        return Error(f"Failed to convert CT from path '{str(ct_path)}' to mu: {tensor.description}")
    tensor = tensor.to(device=device)
    spacing = torch.tensor(volume.GetSpacing(), device=device, dtype=torch.float64)
    if ct_permutation is not None:
        if len(ct_permutation) != 3:
            return Error("Length of ct_permutation must be 3.")
        tensor = tensor.permute(*ct_permutation)
        spacing = spacing[torch.tensor(ct_permutation)]

    logger.info(
        "CT loaded; size = [{} x {} x {}]; spacing = ({}, {}, {})".format(*tensor.size(), *[e.item() for e in spacing]))
    return tensor, spacing


def init_dadg(  #
        *,  #
        config: ExperimentSetConfig,  #
        image_specific_config: ImageSpecificConfigurations,  #
        cache_directory: str,  #
        device: torch.device,  #
        dadg: DirectedAcyclicDataGraph | None = None,  #
) -> Error | None:
    if dadg is None:
        dadg = data_manager()

    # ----- Load the CT series
    assert "ct_path" in config.values
    assert "ct_series_uid" in config.values
    assert isinstance(path_config := config.values["ct_path"], Constant)
    assert isinstance(uid_config := config.values["ct_series_uid"], Constant)
    untruncated_ct_volume, ct_spacing = load_untruncated_ct(path_config.value, uid_config.value, device)

    # -----
    # Initialise the DADG
    # t = Transformation.random_uniform(device=device)
    if isinstance(err := dadg.set_multiple(  #
            device=device,  #
            untruncated_ct_volume=untruncated_ct_volume,  #
            ct_spacing=ct_spacing,  #
            cache_directory=cache_directory,  #
            save_to_cache=False,  #
            source_offset=torch.zeros(2, dtype=torch.float64, device=device),  #
            ap_transformation=Transformation(
                rotation=torch.tensor([0.5 * torch.pi, 0.0, 0.0], dtype=torch.float64, device=device),
                translation=torch.zeros(3, dtype=torch.float64, device=device)),  #
            target_ap_distance=5.0,  #
            saved_transformations=image_specific_config.saved_transformations,  #
            saved_xray_reg_configs=image_specific_config.saved_xray_reg_configs,  #
    ), Error):
        return Error(f"Error setting initial data values: {err.description}")

    # -----
    # Initialise the fixed target image
    if isinstance(c := config.values["xray_path"], Constant) and c.value is None:
        # -----
        # Use a DRR
        if isinstance(err := dadg.set_multiple(  #
                xray_path=None,  #
                regenerate_drr=True,  #
                new_drr_size=torch.Size([1000, 1000]),  #
        ), Error):
            return Error(f"Error setting initial data values: {err.description}")

        if isinstance(err := dadg.add_updater("set_target_image", updaters.set_synthetic_target_image), Error):
            return Error(f"Error adding updater: {err.description}")
    else:
        if isinstance(c := config.values["xray_path"], Cartesian) and len(c.values) > 1 and any(  #
                e is None  #
                for e in c.values  #
        ):
            raise ValueError(f"Cannot run experiments over DRRs and X-ray images.")

        # -----
        # Use X-ray file(s)
        if isinstance(err := dadg.add_updater("set_target_image", updaters.set_xray_target_image), Error):
            return Error(f"Error adding updater: {err.description}")

        if isinstance(err := dadg.add_updater("set_ground_truth", updaters.load_ground_truth), Error):
            return Error(f"Error adding updater: {err.description}")

    # -----
    # Add updaters to the DADG
    if isinstance(err := dadg.add_updater("apply_truncation", updaters.apply_truncation), Error):
        return Error(f"Error adding updater: {err.description}")
    if isinstance(err := dadg.add_updater(  #
            "refresh_image_2d_scale_factor", updaters.refresh_image_2d_scale_factor), Error):
        return Error(f"Error adding updater: {err.description}")
    if isinstance(
            err := dadg.add_updater("refresh_hyperparameter_dependent", updaters.refresh_hyperparameter_dependent),
            Error):
        return Error(f"Error adding updater: {err.description}")
    if False:
        if isinstance(err := dadg.add_updater("refresh_mask_transformation_dependent",
                                              updaters.refresh_mask_transformation_dependent), Error):
            return Error(f"Error adding updater: {err.description}")
        if isinstance(err := dadg.add_updater("project_drr", updaters.project_drr), Error):
            return Error(f"Error adding updater: {err.description}")
    if isinstance(err := dadg.add_updater("load_base_cropping", updaters.load_base_cropping), Error):
        return Error(f"Error adding updater: {err.description}")
    if isinstance(err := dadg.add_updater("combine_croppings", updaters.combine_croppings), Error):
        return Error(f"Error adding updater: {err.description}")
    # Optional
    if False:
        if isinstance(err := dadg.add_updater("truncation_from_h_valid", truncation_percent_for_desired_h_valid),
                Error):
            return Error(f"Error adding updater: {err.description}")
    if True:
        if isinstance(err := dadg.add_updater("project_moving_images", batched.project_moving_images), Error):
            return Error(f"Error adding updater: {err.description}")
        if isinstance(err := dadg.add_updater("apply_sim_metric", batched.apply_sim_metric), Error):
            return Error(f"Error adding updater: {err.description}")
        if isinstance(err := dadg.add_updater("refresh_scaling_images",
                                              dadg_updater(names_returned=["scaling_images", "fixed_images"])(
                                                  batched.refresh_scaling_images)), Error):
            return Error(f"Error adding updater: {err.description}")
    return None
