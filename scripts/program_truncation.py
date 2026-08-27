import argparse
import os
import pathlib
import pprint

import matplotlib

matplotlib.use("QtAgg")

import torch
import yaml

from reg23_experiments.data.structs import Error
from reg23_experiments.experiments.experiment_set_config import Cartesian, Constant, ExperimentSetConfig, Zipped
from reg23_experiments.experiments.helpers import instance_output_directory
from reg23_experiments.experiments.reg_experiment import ExperimentParametrisation, ImageSpecificConfigurations, \
    init_dadg, reg_experiment
from reg23_experiments.experiments.run_experiments import run_experiments
from reg23_experiments.io.command_line import get_string_required
from reg23_experiments.io.serialize import serialize_recursive
from reg23_experiments.io.sitk import DCMSeriesInfo, find_ct_series
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
    # Load all saved transformations and  X-ray configs; these are searched through for ground truth alignments and used
    # for manual X-ray configurations
    image_specific_config = ImageSpecificConfigurations.load()

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
        config = ExperimentSetConfig({  #
            # ----- images
            "ct_path": Constant(ct_path),  #
            "ct_series_uid": Constant(ct_series_uid),  #
            # ----- preprocessing
            "downsample_level": Constant(0),  #
            "truncation_percent": Cartesian([80, 90]),  #
            # ----- cropping
            "cropping_method": Constant("bounding_box"),  #
            "iterations_per_crop_update": Constant(1000),  #
            # ----- scaling
            "apply_scaling": Constant(False),  #
            # ----- similarity & weighting
            "weighting_method": Zipped(["none", "smooth_step", "smooth_step", "smooth_step"]),  #
            "weight_alpha": Zipped([1.0, 1.0, 1.5, 2.0]),  #
            "iterations_per_weight_update": Constant(1000),  #
            "sim_metric": Constant("gradient_correlation"),  #
            # ----- registration
            "starting_distance": Constant(0.5),  # Constant(5.0)
            "sample_count_per_distance": Constant(50),  #
            # ----- PSO config
            "particle_count": Constant(2000),  #
            "particle_initialisation_spread": Constant(0.25),  # Constant(2.5)
            "iteration_count": Constant(5),  #
        })

        # X-ray choice determines the gold standard orientation, which drives h_linear:
        hardcoded_xray_names: list[str] = [  #
            "level_000",  #
            # "level_090",  #
            # "up_000",  #
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
        config = ExperimentSetConfig.deserialize(variables)

        logger.info(f"Filling gaps in experiment output directory '{str(instance_output_dir)}'.")

    # -----
    # Check that all X-rays exist, have ground truth transformations available, and have reg configs available
    if isinstance(c := config.values["xray_path"], Constant):
        if isinstance(err := image_specific_config.check_xray_path(c.value), Error):
            logger.error(f"Invalid X-ray path '{c.value}': {err.description}")
            return
    else:
        assert isinstance(l := config.values["xray_path"], Cartesian)
        for v in l.values:
            if isinstance(err := image_specific_config.check_xray_path(v), Error):
                logger.error(f"Invalid X-ray path '{v}': {err.description}")
                return

    # --------------------------
    # ----- Initialisation -----
    # --------------------------

    err = init_dadg(  #
        config=config,  #
        image_specific_config=image_specific_config,  #
        cache_directory=cache_directory,  #
        device=device  #
    )
    if isinstance(err, Error):
        logger.error(f"Failed to initialize DADG: {err.description}")
        return

    if show:
        run_experiments(  #
            param_constructor=ExperimentParametrisation.dict_constructor,  #
            # experiment=run_experiment,  #
            experiment=lambda conf: reg_experiment(conf, batch_size=250, plot=True),  #
            parametrisation_iterable=(c for c in [next(iter(config.iterable()))]),  # just the first iteration
            output_directory=None,  #
            device=device,  #
            dry_run=False,  #
            throw=True,  #
        )
    else:
        # -----
        # Run experiments, initially just as a dry-run
        for dry_run in [True, False]:
            run_experiments(  #
                param_constructor=ExperimentParametrisation.dict_constructor,  #
                # experiment=run_experiment,  #
                experiment=lambda conf: reg_experiment(conf, batch_size=250),  #
                parametrisation_iterable=config.iterable(space_sample_count=64),  #
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
