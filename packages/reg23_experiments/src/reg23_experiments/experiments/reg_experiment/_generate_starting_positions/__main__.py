import argparse
import logging
import pathlib
import pprint

import pydicom
import torch

from reg23_experiments.data.save_data.transformations import TransformationSaveManager
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.io.command_line import get_string_required
from reg23_experiments.io.image import read_dicom
from reg23_experiments.io.serialize import serialize_recursive
from reg23_experiments.io.sitk import DCMSeriesInfo, find_ct_series
from reg23_experiments.utils import logs_setup


def acquire_ct_series_uid(ct_path: str) -> str | Error:
    series: dict[str, DCMSeriesInfo] | Error = find_ct_series(ct_path)
    if isinstance(series, Error):
        return series
    if not series:
        return Error(f"No CT series found at path '{ct_path}'.")
    if len(series) == 1:
        return next(iter(series))
    return get_string_required(  #
        f"Please choose one of the following CT series:\n"
        f"{"\n".join(f"{k}:\n\t{pprint.pformat(serialize_recursive(v))}\n" for k, v in series.items())}",  #
        lambda k: None if k in series else Error(f"String '{k}' does not name a series.")  #
    )


def main(  #
    *,  #
    ct_path: str,  #
    xray_path: str,  #
    transformation_name: str,  #
    distance: float,  #
    count: int,  #
    seed: int | None,  #
    output_dir: str,  #
) -> int:
    output_dir = pathlib.Path(output_dir)

    if count < 1:
        logger.error(f"Count must be at least 1; got {count}.")
        return 1
    if output_dir.exists() and not output_dir.is_dir():
        logger.error(f"Output path '{str(output_dir)}' isn't a directory.")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    xray_uid: str = read_dicom(xray_path)["uid"]

    output_path = output_dir / f"{xray_uid}.pt"

    ct_series_uid: str | Error = acquire_ct_series_uid(ct_path)
    if isinstance(ct_series_uid, Error):
        logger.error(f"Failed to acquire CT series UID from path '{ct_path}': {ct_series_uid.description}")
        return 1

    transformation_gt: Transformation | Error = TransformationSaveManager(  #
        pathlib.Path("data/app_transformation_save_data")  #
    ).get_transformation(  #
        source_uid=ct_series_uid,  #
        destination_uid=xray_uid,  #
        name=transformation_name,  #
    )
    if isinstance(transformation_gt, Error):
        logger.error(f"Failed to get transformation '{transformation_name}': {transformation_gt.description}")
        return 1

    if seed is not None:
        torch.manual_seed(seed)

    # Each row is a transformation: the axis-angle rotation, followed by the translation
    starting_positions: torch.Tensor = torch.stack([  #
        transformation_gt.with_random_offset_at_distance(distance).vectorised()  #
        for _ in range(count)  #
    ])  # size = (count, 6)

    torch.save(starting_positions, output_path)
    logger.info(f"Saved {count} starting positions at distance {distance} to '{str(output_path)}'.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(  #
        description="Generate random starting positions at a fixed distance from a saved transformation, and save "
                    "them to a file to be loaded by `reg_experiment`.",  #
        epilog="")

    parser.add_argument("-p", "--ct-path", type=str, required=True,  #
        help="Give a path to a directory of .dcm files containing the CT data.")
    parser.add_argument("-x", "--xray-path", type=str, required=True,  #
        help="Give a path to a DICOM file containing the X-ray image.")
    parser.add_argument("-n", "--transformation-name", type=str, default="gold_standard",  #
        help="The name of the saved transformation to offset from; default = gold_standard.")
    parser.add_argument("-d", "--distance", type=float, required=True,  #
        help="The distance from the saved transformation at which to generate starting positions.")
    parser.add_argument("-c", "--count", type=int, required=True,  #
        help="The number of starting positions to generate; must be at least the `sample_count_per_distance` of the "
             "experiment.")
    parser.add_argument("-s", "--seed", type=int, default=None,  #
        help="Seed for the random number generator. If not provided, the positions will differ every run.")
    parser.add_argument("-o", "--output-dir", type=str, required=True,  #
        help="The directory in which to save the starting positions.")
    parser.add_argument("-l", "--log-level", type=str, default=None,  #
        help=f"Set the log level; options: [{", ".join(logging._nameToLevel.keys())}]; default = INFO.")

    args = parser.parse_args()

    logger = logs_setup.setup_logger(level=args.log_level)

    exit(main(  #
        ct_path=args.ct_path,  #
        xray_path=args.xray_path,  #
        transformation_name=args.transformation_name,  #
        distance=args.distance,  #
        count=args.count,  #
        seed=args.seed,  #
        output_dir=args.output_dir,  #
    ))
