import argparse
import os
import pathlib
import pprint
from typing import Any, Sequence

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import yaml

from reg23_experiments.data.structs import Cropping, Error, Transformation
from reg23_experiments.data.transformation_save_data import TransformationSaveData
from reg23_experiments.data.xray_reg_save_data import XRayRegSaveData
from reg23_experiments.experiments.config import Cartesian, Constant, ExperimentConfig, Zipped
from reg23_experiments.experiments.dadg_updaters import batched
from reg23_experiments.experiments.dadg_updaters import drr_reg as updaters
from reg23_experiments.experiments.helpers import instance_output_directory
from reg23_experiments.experiments.reg_experiment import exp_config_from_dict, run_experiment
from reg23_experiments.experiments.run import experiments_hybrid
from reg23_experiments.io.command_line import get_string_required
from reg23_experiments.io.image import XrayDICOM, read_dicom
from reg23_experiments.io.save_data import load_latest_save
from reg23_experiments.io.serialize import serialize_recursive
from reg23_experiments.io.sitk import DCMSeriesInfo, find_ct_series, load_ct_series
from reg23_experiments.ops import geometry
from reg23_experiments.ops.ct import convert_ct_to_mu_sitk
from reg23_experiments.ops.data_manager import dadg_updater, data_manager
from reg23_experiments.utils import logs_setup, pushover


def acquire_ct_series_uid(ct_path: pathlib.Path) -> str | Error:
    series: dict[str, DCMSeriesInfo] | Error = find_ct_series(ct_path)
    if isinstance(series, Error):
        raise Exception(f"Failed to open CT from path '{ct_path}': {series.description}")
    if not series:
        return Error(f"No CT series found at path '{str(ct_path)}'.")
    if len(series) == 1:
        ct_series_uid = next(iter(series))
    else:
        ct_series_uid = get_string_required(  #
            f"Please choose one of the following CT series:\n"
            f"{"\n".join(f"{k}:\n\t{pprint.pformat(serialize_recursive(v))}\n" for k, v in series.items())}",  #
            lambda k: None if k in series else Error(f"String '{k}' does not name a series.")  #
        )
    return ct_series_uid


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
        show: bool = False,  #
        fill_gaps: str | None = None,  #
        name: str | None = None,  #
):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if xray_path is not None:
        xray_path = pathlib.Path(xray_path)

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

    # -----------------------------
    # ----- Experiment config -----
    # -----------------------------

    if fill_gaps is None:
        # -----
        # Look for series at the CT path, prompting the user to choose one series if multiple are found
        res: str | Error = acquire_ct_series_uid(pathlib.Path(ct_path))
        if isinstance(res, Error):
            raise Exception(f"Failed to find CT series: {res.description}")
        ct_series_uid = res

        # ----------------------------------
        # - Hardcoded script configuration -
        # ----------------------------------
        config = ExperimentConfig({  #
            # ----- images
            "ct_path": Constant(ct_path),  #
            "ct_series_uid": Constant(ct_series_uid),  #
            # ----- preprocessing
            "downsample_level": Cartesian([1, 2, 3]),  #
            "truncation_percent": Constant(85),  #
            # ----- cropping
            "cropping_method": Constant("bounding_box"),  #
            "iterations_per_crop_update": Constant(1000),  #
            # ----- scaling
            "apply_scaling": Constant(False),  #
            # ----- similarity & weighting
            "apply_weighting": Cartesian([False, True]),  #
            "weight_alpha": Constant(1.0),  #
            "iterations_per_weight_update": Constant(1000),  #
            "sim_metric": Constant("gradient_correlation"), #
            # ----- registration
            "starting_distance": Cartesian([15.0, 17.5, 20.0]),  # Constant(5.0)
            "sample_count_per_distance": Constant(10),  #
            # ----- PSO config
            "particle_count": Constant(2000),  #
            "particle_initialisation_spread": Constant(7.5),  # Constant(2.5)
            "iteration_count": Constant(4),  #
        })

        # X-ray choice determines the gold standard orientation, which drives h_linear:
        hardcoded_xray_names: list[str] = [  #
            "level_000",  #
            # "level_090",  #
            "up_000",  #
            # "up_090",  #
            "down_000",  #
            # "down_090",  #
        ]

        # -----
        # Set the X-ray path(s) depending on if a directory or filename is passed
        if xray_path is None or xray_path.is_file():
            config.values["xray_path"] = Constant(xray_path)
        elif xray_path.is_dir():
            if len(hardcoded_xray_names) == 1:
                config.values["xray_path"] = Constant(str(xray_path / hardcoded_xray_names[0]))
            else:
                config.values["xray_path"] = Cartesian([str(xray_path / name) for name in hardcoded_xray_names])

        if not show:
            instance_output_dir: pathlib.Path = instance_output_directory(data_output_dir, name)

            with open(instance_output_dir / "variables.txt", 'w') as file:
                yaml.safe_dump(config.serialize(), file, sort_keys=False)  # very important to preserve order of keys
    else:
        assert not show

        instance_output_dir = pathlib.Path(data_output_dir) / fill_gaps
        if not instance_output_dir.is_dir():
            raise FileNotFoundError(f"Directory in which to fill gaps '{str(instance_output_dir)}' not found.")

        variables_path = instance_output_dir / "variables.txt"
        if not variables_path.is_file():
            raise FileNotFoundError(f"File '{str(variables_path)}' not found; cannot fill gaps.")

        variables = yaml.safe_load(variables_path.read_text())
        assert isinstance(variables, dict)
        config = ExperimentConfig.deserialize(variables)

        logger.info(f"Filling gaps in experiment output directory '{str(instance_output_dir)}'.")

    # -----
    # Check that all X-rays exist, have ground truth transformations available, and have reg configs available
    def check_xray_path(p: str | pathlib.Path) -> Error | None:
        p = pathlib.Path(p)
        if not p.is_file():
            return Error(f"X-ray file '{str(p)}' doesn't exist.")
        try:
            dicom: XrayDICOM = read_dicom(p)
        except Exception as e:
            return Error(f"Failed to read X-ray file: {e}")
        idx = (dicom["uid"], "gold_standard")
        try:
            saved_transformations.loc[idx]
        except KeyError:
            return Error(f"No ground truth saved for X-ray '{str(p)}' with UID '{dicom["uid"]}'.")
        idx = dicom["uid"]
        try:
            saved_xray_reg_configs.loc[idx]
        except KeyError:
            return Error(f"No reg config saved for X-ray '{str(p)}' with UID '{dicom["uid"]}'.")
        return None

    if isinstance(c := config.values["xray_path"], Constant):
        if isinstance(err := check_xray_path(c.value), Error):
            logger.error(f"Invalid X-ray path '{c.value}': {err.description}")
            return
    else:
        assert isinstance(l := config.values["xray_path"], Cartesian)
        for v in l.values:
            if isinstance(err := check_xray_path(v), Error):
                logger.error(f"Invalid X-ray path '{v}': {err.description}")
                return

    # --------------------------
    # ----- Initialisation -----
    # --------------------------

    # ----- Load the CT series
    assert "ct_path" in config.values
    assert "ct_series_uid" in config.values
    assert isinstance(path_config := config.values["ct_path"], Constant)
    assert isinstance(uid_config := config.values["ct_series_uid"], Constant)
    untruncated_ct_volume, ct_spacing = load_untruncated_ct(path_config.value, uid_config.value, device)

    # -----
    # Initialise the DADG
    # t = Transformation.random_uniform(device=device)
    if isinstance(err := data_manager().set_multiple(  #
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
            saved_transformations=saved_transformations,  #
            saved_xray_reg_configs=saved_xray_reg_configs,  #
    ), Error):
        logger.error(f"Error setting initial data values: {err.description}")
        return

    # -----
    # Initialise the fixed target image
    if isinstance(c := config.values["xray_path"], Constant) and c.value is None:
        # -----
        # Use a DRR
        if isinstance(err := data_manager().set_multiple(  #
                xray_path=None,  #
                regenerate_drr=True,  #
                new_drr_size=torch.Size([1000, 1000]),  #
        ), Error):
            logger.error(f"Error setting initial data values: {err.description}")
            return

        if isinstance(err := data_manager().add_updater("set_target_image", updaters.set_synthetic_target_image),
                      Error):
            logger.error(f"Error adding updater: {err.description}")
            return
    else:
        if isinstance(c := config.values["xray_path"], Cartesian) and len(c.values) > 1 and any(  #
                e is None  #
                for e in c.values  #
        ):
            raise ValueError(f"Cannot run experiments over DRRs and X-ray images.")

        # -----
        # Use X-ray file(s)
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
    if False:
        if isinstance(
                err := data_manager().add_updater("truncation_from_h_valid", truncation_percent_for_desired_h_valid),
                Error):
            logger.error(f"Error adding updater: {err.description}")
            return
    if True:
        if isinstance(err := data_manager().add_updater("project_moving_images", batched.project_moving_images), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
        if isinstance(err := data_manager().add_updater("apply_sim_metric", batched.apply_sim_metric), Error):
            logger.error(f"Error adding updater: {err.description}")
            return
        if isinstance(err := data_manager().add_updater("refresh_scaling_images",
                                                        dadg_updater(names_returned=["scaling_images", "fixed_images"])(
                                                            batched.refresh_scaling_images)), Error):
            logger.error(f"Error adding updater: {err.description}")
            return

    if show:
        experiments_hybrid(  #
            param_constructor=exp_config_from_dict,  #
            # experiment=run_experiment,  #
            experiment=lambda conf, dev, pos, dry: run_experiment(conf, dev, pos, dry, 250, plot=True),  #
            config_iterable=(c for c in [next(iter(config.iterable()))]),  # just the first iteration
            output_directory=None,  #
            device=device,  #
            dry_run=False,  #
            throw=True,  #
        )
    else:
        # -----
        # Run experiments, initially just as a dry-run
        for dry_run in [True, False]:
            experiments_hybrid(  #
                param_constructor=exp_config_from_dict,  #
                # experiment=run_experiment,  #
                experiment=lambda conf, dev, pos, dry: run_experiment(conf, dev, pos, dry, 250),  #
                config_iterable=config.iterable(space_sample_count=64),  #
                output_directory=instance_output_dir,  #
                device=device,  #
                dry_run=dry_run,  #
                throw=dry_run,  #
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
    parser.add_argument("-n", "--notify", action="store_true", help="Send notification on completion.")
    parser.add_argument("-s", "--show", action="store_true", help="Show images at the G.T. alignment.")
    parser.add_argument("-o", "--data-output-dir", type=str, default="experimental_results/program_truncation",
                        help="Directory in which to save output data.")
    parser.add_argument("-f", "--fill-gaps", type=str, default=None,
                        help="Give a path to an existing results directory and run all experiments specified by "
                             "variables.txt whose results are not present.")
    parser.add_argument("--name", type=str, default=None,
                        help="Name to give the experiment; this will just be used in the name of the output directory.")
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
             data_output_dir=args.data_output_dir, show=args.show, fill_gaps=args.fill_gaps, name=args.name)
        if args.notify:
            pushover.send_notification(__file__, "Script finished.")
    except Exception as e:
        if args.notify:
            pushover.send_notification(__file__, "Script raised exception: {}.".format(e))
        raise e
