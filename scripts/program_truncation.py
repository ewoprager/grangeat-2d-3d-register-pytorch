import argparse
import os
import pathlib
import pprint
from typing import Any, Sequence

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

from reg23_experiments.data.structs import Cropping, Error, LinearRange, Transformation
from reg23_experiments.data.transformation_save_data import TransformationSaveData
from reg23_experiments.data.xray_reg_save_data import XRayRegSaveData
from reg23_experiments.experiments.dadg_updaters import drr_reg as updaters
from reg23_experiments.experiments.helpers import instance_output_directory
from reg23_experiments.experiments.reg_experiment2 import exp_config_from_dict, run_experiment
from reg23_experiments.experiments.registration import RegConfig, run_reg
from reg23_experiments.experiments.run import experiments_sobol
from reg23_experiments.io.command_line import get_string_required
from reg23_experiments.io.image import XrayDICOM, read_dicom
from reg23_experiments.io.save_data import load_latest_save
from reg23_experiments.io.serialize import serialize_recursive
from reg23_experiments.io.sitk import DCMSeriesInfo, find_ct_series, load_ct_series
from reg23_experiments.ops import geometry, similarity_metric
from reg23_experiments.ops.ct import convert_ct_to_mu_sitk
from reg23_experiments.ops.data_manager import args_from_dadg, dadg_updater, data_manager
from reg23_experiments.ops.optimisation import mapping_parameters_to_transformation, \
    mapping_transformation_to_parameters, random_parameters_at_distance
from reg23_experiments.utils import logs_setup, pushover


def load_untruncated_ct(  #
        ct_path: pathlib.Path,  #
        device: torch.device,  #
        ct_permutation: Sequence[int] | None = None  #
) -> tuple[torch.Tensor, torch.Tensor, str] | Error:
    series: dict[str, DCMSeriesInfo] | Error = find_ct_series(ct_path)
    if isinstance(series, Error):
        raise Exception(f"Failed to open CT from path '{ct_path}': {series.description}")
    if not series:
        return Error(f"No CT series found at path '{str(ct_path)}'.")
    if len(series) == 1:
        key = next(iter(series))
    else:
        key = get_string_required(  #
            f"Please choose one of the following CT series:\n"
            f"{"\n".join(f"{k}:\n\t{pprint.pformat(serialize_recursive(v))}\n" for k, v in series.items())}",  #
            lambda k: None if k in series else Error(f"String '{k}' does not name a series.")  #
        )
    volume: sitk.Image | Error = load_ct_series(ct_path, key)
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

    return tensor, spacing, key


@dadg_updater(names_returned=["transformation_gt"])
def load_ground_truth(  #
        *,  #
        saved_transformations: pd.DataFrame,  #
        xray_sop_instance_uid: str,  #
        device: torch.device  #
) -> dict[str, Any]:
    idx = (xray_sop_instance_uid, "gold_standard")
    try:
        row = saved_transformations.loc[idx]
    except KeyError:
        return {"transformation_gt": None}
    return {  #
        "transformation_gt": Transformation.from_vector(  #
            torch.tensor([row[f"x{i}"] for i in range(6)], device=device, dtype=torch.float64)  #
        )  #
    }


@dadg_updater(names_returned=["base_cropping", "target_flipped"])
def load_base_cropping(  #
        *,  #
        saved_xray_reg_configs: pd.DataFrame,  #
        xray_sop_instance_uid: str,  #
) -> dict[str, Any]:
    try:
        row = saved_xray_reg_configs.loc[xray_sop_instance_uid]
    except KeyError:
        return {"base_cropping": None}
    return {  #
        "base_cropping": Cropping(  #
            left=row["crop_left"],  #
            right=row["crop_right"],  #
            top=row["crop_top"],  #
            bottom=row["crop_bottom"],  #
        ),  #
        "target_flipped": row["horizontal_flip"],  #
    }


@dadg_updater(names_returned=["cropping"])
def combine_croppings(  #
        *,  #
        base_cropping: Cropping | None,  #
        further_cropping: Cropping | None,  #
) -> dict[str, Any]:
    if base_cropping is None:
        if further_cropping is None:
            return {"cropping": None}
        else:
            return {"cropping": further_cropping}
    else:
        if further_cropping is None:
            return {"cropping": base_cropping}
        else:
            return {"cropping": Cropping.intersect(base_cropping, further_cropping)}


@dadg_updater(names_returned=["truncation_percent"])
def truncation_percent_for_desired_h_valid(  #
        *,  #
        transformation_gt: Transformation | None,  #
        untruncated_ct_volume: torch.Tensor,  #
        ct_spacing: torch.Tensor,  #
        desired_h_valid: float,  #
) -> dict[str, Any]:
    if transformation_gt is None:
        raise Exception("Need transformation gold standard for h_valid")
    theta = abs(
        geometry.axis_angle_extract_axis(transformation_gt.rotation, torch.tensor([1.0, 0.0, 0.0])) - 0.5 * np.pi)
    l = ct_spacing[1].item() * float(untruncated_ct_volume.size()[1])
    full_height = ct_spacing[2].item() * float(untruncated_ct_volume.size()[0])
    h = (desired_h_valid + l * np.sin(theta)) / np.cos(theta)
    truncation_percent = min(98, max(0, round(100.0 * (1.0 - h / full_height))))
    return {"truncation_percent": truncation_percent}


def main(  #
        *,  #
        cache_directory: str,  #
        ct_path: str,  #
        xray_path: str | pathlib.Path | None,  #
        data_output_dir: str | pathlib.Path,  #
        show: bool = False  #
):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if xray_path is not None:
        xray_path = pathlib.Path(xray_path)

    # -----
    # Load the CT data, prompting the user to choose a series if multiple are found
    res = load_untruncated_ct(pathlib.Path(ct_path), device)
    if isinstance(res, Error):
        raise Exception(f"Failed to load CT: {res.description}")
    untruncated_ct_volume, ct_spacing, ct_series_uid = res

    # -----
    # Load all saved transformations; these are searched through for ground truth alignments
    res: tuple[pathlib.Path, TransformationSaveData, int] | Error = load_latest_save(  #
        TransformationSaveData,  #
        save_directory=pathlib.Path("data/app_transformation_save_data")  #
    )
    if isinstance(res, Error):
        raise RuntimeError(f"Failed to load saved transformation: {res.description}")
    _, transformation_save_data, _ = res
    saved_transformations: pd.DataFrame = transformation_save_data.get_data()
    logger.info(f"Saved transformation data:\n{saved_transformations.to_string()}")

    # -----
    # Load all saved X-ray configs; these are used for manual X-ray configurations
    res: tuple[pathlib.Path, XRayRegSaveData, int] | Error = load_latest_save(  #
        XRayRegSaveData,  #
        save_directory=pathlib.Path("data/xray_reg_save_data")  #
    )
    if isinstance(res, Error):
        raise RuntimeError(f"Failed to load saved X-ray reg configs: {res.description}")
    _, xray_reg_save_data, _ = res
    saved_xray_reg_configs: pd.DataFrame = xray_reg_save_data.get_data()
    logger.info(f"Saved X-ray reg configs:\n{saved_xray_reg_configs.to_string()}")

    # -----
    # Initialise the DADG
    if isinstance(err := data_manager().set_multiple(  #
            device=device,  #
            untruncated_ct_volume=untruncated_ct_volume,  #
            ct_spacing=ct_spacing,  #
            ct_series_uid=ct_series_uid,  #
            cache_directory=cache_directory,  #
            save_to_cache=False,  #
            # truncation_percent=0,  #
            desired_h_valid=20.0,  #
            further_cropping=None,  #
            source_offset=torch.zeros(2, dtype=torch.float64, device=device),  #
            downsample_level=0,  #
            ap_transformation=Transformation(
                rotation=torch.tensor([0.5 * torch.pi, 0.0, 0.0], dtype=torch.float64, device=device),
                translation=torch.zeros(3, dtype=torch.float64, device=device)),  #
            target_ap_distance=5.0,  #
            current_transformation=Transformation.random_uniform(device=device),  #
            mask_transformation=None,  #
            saved_transformations=saved_transformations,  #
            saved_xray_reg_configs=saved_xray_reg_configs,  #
    ), Error):
        logger.error(f"Error setting initial data values: {err.description}")
        return

    # -----
    # Initialise the fixed target image
    if xray_path is None:
        # -----
        # Use a DRR
        if isinstance(err := data_manager().set_multiple(  #
                xray_path=None,  #
                regenerate_drr=True,  #
                new_drr_size=torch.Size([1000, 1000]),  #
                target_ap_distance=5.0,  #
        ), Error):
            logger.error(f"Error setting initial data values: {err.description}")
            return

        if isinstance(err := data_manager().add_updater("set_target_image", updaters.set_synthetic_target_image),
                      Error):
            logger.error(f"Error adding updater: {err.description}")
            return
    elif xray_path.is_dir():
        # -----
        # Use a directory of X-ray images
        if isinstance(err := data_manager().add_updater("set_target_image", updaters.set_xray_target_image), Error):
            logger.error(f"Error adding updater: {err.description}")
            return

        if isinstance(err := data_manager().add_updater("set_ground_truth", load_ground_truth), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
    else:
        # -----
        # Use an X-ray image
        if not xray_path.is_file():
            raise Exception(f"X-ray file '{str(xray_path)}' not found.")

        if isinstance(err := data_manager().set("xray_path", xray_path), Error):
            logger.error(f"Error setting initial data values: {err.description}")
            return
        if isinstance(err := data_manager().add_updater("set_target_image", updaters.set_xray_target_image), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
        if isinstance(err := data_manager().add_updater("set_ground_truth", load_ground_truth), Error):
            logger.error(f"Error adding updater: {err.description}")
            return

    # -----
    # Add updaters to the DADG
    if isinstance(err := data_manager().add_updater("apply_truncation", updaters.apply_truncation), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater(  #
            "refresh_image_2d_scale_factor", updaters.refresh_image_2d_scale_factor), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater("refresh_hyperparameter_dependent",
                                                    updaters.refresh_hyperparameter_dependent), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if False:
        if isinstance(err := data_manager().add_updater("refresh_mask_transformation_dependent",
                                                        updaters.refresh_mask_transformation_dependent), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
        if isinstance(err := data_manager().add_updater("project_drr", updaters.project_drr), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
    if isinstance(err := data_manager().add_updater("load_base_cropping", load_base_cropping), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    if isinstance(err := data_manager().add_updater("combine_croppings", combine_croppings), Error):
        logger.error(f"Error adding updater: {err.description}")
        return
    # Optional
    if isinstance(err := data_manager().add_updater("truncation_from_h_valid", truncation_percent_for_desired_h_valid),
                  Error):
        logger.error(f"Error adding updater: {err.description}")
        return

    # ----------------------------------
    # - Hardcoded script configuration -
    # ----------------------------------
    constants: dict[str, Any] = {  #
        # ExperimentConfig
        "ct_path": ct_path,  #
        "xray_path": xray_path,  #
        "ct_series_uid": data_manager().get("ct_series_uid"),  #
        "downsample_level": 1,  #
        # "truncation_percent": 80,  #
        "desired_h_valid": 60.0,  #
        # "cropping": "full_depth_drr",  #
        # "crop_expand": 0.0,  #
        "crop_min_size": 0.01,  #
        "weight_alpha": 0.0,  #
        # "mask": "None",  #
        "sim_metric": "zncc",  #
        "starting_distance": 3.0,  #
        "sample_count_per_distance": 10,  #
        # RegConfig
        "particle_count": 2000,  #
        "particle_initialisation_spread": 5.0,  #
        "iteration_count": 6,  #
    }
    # X-ray choice determines the gold standard orientation, which drives h_linear:
    hardcoded_xray_names: list[str] = [  #
        "level_000",  #
        # "level_090",  #
        # "up_000",  #
        # "up_090",  #
        # "down_000",  #
        # "down_090",  #
    ]
    params_to_vary: dict[str, list | LinearRange] = {  #
        # "desired_h_valid": [float(e) for e in np.linspace(20.0, 33.0, 16)],  #
        "desired_h_valid": LinearRange(10.0, 60.0),  #
        # "crop_expand": LinearRange(0.0, 30.0),  #
    }
    # ----------------------------------

    # -----
    # Setting the X-ray path(s) if a directory is passed
    if xray_path is not None and xray_path.is_dir():
        # Check that all X-rays exist, have ground truth transformations available, and have reg configs available
        for name in hardcoded_xray_names:
            path: pathlib.Path = xray_path / name
            if not path.is_file():
                logger.error(f"X-ray file '{str(path)}' doesn't exist.")
                return
            try:
                dicom: XrayDICOM = read_dicom(path)
            except Exception as e:
                logger.error(f"Failed to read X-ray file: {e}")
                return
            idx = (dicom["uid"], "gold_standard")
            try:
                saved_transformations.loc[idx]
            except KeyError:
                logger.error(f"No ground truth saved for X-ray '{str(path)}' with UID '{dicom["uid"]}'.")
                return
            idx = dicom["uid"]
            try:
                saved_xray_reg_configs.loc[idx]
            except KeyError:
                logger.error(f"No reg config saved for X-ray '{str(path)}' with UID '{dicom["uid"]}'.")
                return
        if len(hardcoded_xray_names) == 1:
            constants["xray_path"] = str(xray_path / hardcoded_xray_names[0])
        else:
            params_to_vary["xray_path"] = [str(xray_path / name) for name in hardcoded_xray_names]

    # Remove varying variables from the constants dict
    for key in params_to_vary:
        if key in constants:
            constants.pop(key)

    if show:
        # -----
        # Display images for debugging
        plt.ion()  # figures are non-blocking
        plt.show()
        fig, axes = plt.subplots(1, 4)
        # -----
        # Set the current transformation to the ground truth if it exists
        data_manager().set("xray_path", "/home/eprager/Documents/Datasets/3DP Head 2/X-ray/down_090")

        transformation_gt: Transformation | None | Error = data_manager().get("transformation_gt")
        if transformation_gt is None or isinstance(transformation_gt, Error):
            raise RuntimeError(f"No ground truth available"
                               f"{"." if transformation_gt is None else f": {transformation_gt.description}"}")
        parameters_gt = mapping_transformation_to_parameters(transformation_gt)
        starting_params = random_parameters_at_distance(parameters_gt, constants["starting_distance"])

        data_manager().set("current_transformation", transformation_gt)
        if "downsample_level" in constants:
            data_manager().set("downsample_level", constants["downsample_level"])
        image_2d_full: torch.Tensor | Error = data_manager().get("image_2d_full")
        if isinstance(image_2d_full, Error):
            raise RuntimeError(f"Error getting image_2d_full: {image_2d_full.description}")
        axes[0].imshow(image_2d_full.cpu().numpy())
        axes[0].set_title("original target")
        fixed_image: torch.Tensor | Error = data_manager().get("fixed_image")
        if isinstance(fixed_image, Error):
            raise RuntimeError(f"Error getting fixed image: {fixed_image.description}")
        axes[1].imshow(fixed_image.cpu().numpy())
        axes[1].set_title("fixed image")
        moving_image: torch.Tensor | Error = data_manager().get("moving_image")
        if isinstance(moving_image, Error):
            raise RuntimeError(f"Error getting moving image: {moving_image.description}")
        axes[2].imshow(moving_image.cpu().numpy())
        axes[2].set_title("moving image at G.T.")
        data_manager().set("mask_transformation", data_manager().get("current_transformation"))
        mask: torch.Tensor | Error = data_manager().get("mask")
        if isinstance(mask, Error):
            raise RuntimeError(f"Error getting mask: {mask.description}")
        axes[3].imshow(mask.cpu().numpy())
        axes[3].set_title("mask at G.T.")
        logger.info(f"ZNCC at G.T. with masking = "
                    f"{-similarity_metric.weighted_local_ncc(moving_image, fixed_image, mask, kernel_size=8)}")
        plt.draw()
        plt.pause(0.1)

        data_manager().set("current_transformation", mapping_parameters_to_transformation(starting_params))
        data_manager().set("desired_h_valid", constants["desired_h_valid"])
        if "cropping" in constants:
            if constants["cropping"] == "None":
                cropping: Cropping | None = None
            elif constants["cropping"] == "nonzero_drr":
                cropping: Cropping | None = args_from_dadg()(geometry.get_crop_nonzero_drr)()
            elif constants["cropping"] == "full_depth_drr":
                cropping: Cropping | None = args_from_dadg()(geometry.get_crop_full_depth_drr)()
            else:
                raise ValueError(f"Unknown cropping technique '{constants["cropping"]}'.")
            if isinstance(cropping, Error):
                raise RuntimeError(f"Failed to set crop: {cropping.description}")
            image: torch.Tensor | Error = data_manager().get("image_2d_full")
            if isinstance(image, Error):
                raise Exception(f"Failed to get image_2d_full: {image.description}")
            spacing: torch.Tensor | Error = data_manager().get("image_2d_full_spacing")
            if isinstance(spacing, Error):
                raise Exception(f"Failed to get image_2d_full_spacing: {spacing.description}")
            spacing = spacing.cpu()
            if cropping is not None:
                if cropping.is_collapsed(constants["crop_min_size"]):
                    cropping = cropping.uncollapse(constants["crop_min_size"])
                cropping = cropping.expand_mm(constants["crop_expand"], image_size=image.size(), image_spacing=spacing)
                # expand could be negative, so checking again for collapse
                if cropping.is_collapsed(constants["crop_min_size"]):
                    cropping = cropping.uncollapse(constants["crop_min_size"])
            data_manager().set("further_cropping", cropping, check_equality=True)

        def objective_function(parameters: torch.Tensor) -> torch.Tensor:
            data_manager().set("current_transformation",
                               mapping_parameters_to_transformation(parameters.to(dtype=torch.float64)))
            _moving_image: torch.Tensor | Error = data_manager().get("moving_image")
            _fixed_image: torch.Tensor | Error = data_manager().get("fixed_image")
            return -similarity_metric.ncc(_moving_image, _fixed_image)

        res: torch.Tensor = run_reg(  #
            obj_fun=objective_function,  #
            starting_params=starting_params, config=RegConfig(  #
                particle_count=constants["particle_count"],  #
                particle_initialisation_spread=constants["particle_initialisation_spread"],  #
                iteration_count=constants["iteration_count"],  #
            ),  #
            device=device,  #
            plot="mask")  # size = (iteration count, dimensionality + 1)
        logger.info(f"Result: {res}")
        plt.ioff()  # figures are blocking
        fig, axes = plt.subplots()
        distances = torch.linalg.vector_norm(res[:, :6] - parameters_gt.unsqueeze(0), dim=1).cpu().numpy()
        axes.plot(distances)
        axes.set_xlabel("iteration")
        axes.set_ylabel("distance from G.T.")
        plt.show()
        return

    instance_output_dir: pathlib.Path = instance_output_directory(data_output_dir)

    with open(instance_output_dir / "variables.txt", 'w') as file:
        yaml.safe_dump({  #
            "constants": constants,  #
            "variables": params_to_vary,  #
        }, file)

    # -----
    # Perform a dry-run of the experiments, setting the parameters to vary
    experiments_sobol(  #
        m=1,  #
        param_constructor=exp_config_from_dict,  #
        experiment=lambda conf, dev, pos, dry: run_experiment(conf, dev, pos, dry, 4),  #
        params_to_vary=params_to_vary,  #
        output_directory=instance_output_dir,  #
        constants=constants,  #
        device=device,  #
        dry_run=True,  #
    )

    # -----
    # Run experiments, setting the parameters to vary
    experiments_sobol(  #
        m=1,  #
        param_constructor=exp_config_from_dict,  #
        experiment=lambda conf, dev, pos, dry: run_experiment(conf, dev, pos, dry, 4),  #
        params_to_vary=params_to_vary,  #
        output_directory=instance_output_dir,  #
        constants=constants,  #
        device=device,  #
    )


if __name__ == "__main__":
    # set up logger
    logger = logs_setup.setup_logger()

    # parse arguments
    parser = argparse.ArgumentParser(description="", epilog="")
    parser.add_argument("-c", "--cache-directory", type=str, default="cache",
                        help="Set the directory where data that is expensive to calculate will be saved. The default "
                             "is 'cache'.")
    parser.add_argument("-p", "--ct-path", type=str, required=True,
                        help="Give a path to a .nrrd file, .nii file or directory of .dcm files containing CT data to "
                             "process. If not provided, some simple synthetic data will be used instead - note that "
                             "in this case, data will not be saved to the cache.")
    parser.add_argument("-x", "--xray-path", type=str, default=None,
                        help="Give a path to a DICOM file containing an X-ray image to register the CT image to. If "
                             "this is provided, the X-ray will by used instead of any DRR.")
    parser.add_argument("-d", "--xray-dir", type=str, default=None,
                        help="Give a path to directory of DICOM X-ray images to register the CT image to. If "
                             "this is provided, the X-rays will by used instead of any DRR.")
    # parser.add_argument("-i", "--no-load", action='store_true',
    #                     help="Do not load any pre-calculated data from the cache.")
    # parser.add_argument(
    #     "-r", "--regenerate-drr", action='store_true',
    #     help="Regenerate the DRR through the 3D data, regardless of whether a DRR has been cached.")
    # parser.add_argument("-n", "--no-save", action='store_true', help="Do not save any data to the cache.")
    parser.add_argument("-n", "--notify", action="store_true", help="Send notification on completion.")
    parser.add_argument("-s", "--show", action="store_true", help="Show images at the G.T. alignment.")
    parser.add_argument("-o", "--data-output-dir", type=str, default="experimental_results/program_truncation",
                        help="Directory in which to save output data.")
    args = parser.parse_args()

    if args.xray_path is None:
        if args.xray_dir is None:
            xray = None
        else:
            xray = pathlib.Path(args.xray_dir)
            if not xray.is_dir():
                logger.error(f"X-ray directory '{str(xray)}' doesn't exist.")
                exit(1)
    else:
        if args.xray_dir is None:
            xray = pathlib.Path(args.xray_path)
            if not xray.is_file():
                logger.error(f"X-ray file '{str(xray)}' doesn't exist.")
                exit(1)
        else:
            logger.error(f"Cannot provide both an X-ray directory and an X-ray file.")
            exit(1)

    # create cache directory
    if not os.path.exists(args.cache_directory):
        os.makedirs(args.cache_directory)

    try:
        main(cache_directory=args.cache_directory, ct_path=args.ct_path, xray_path=xray,
             data_output_dir=args.data_output_dir, show=args.show)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
