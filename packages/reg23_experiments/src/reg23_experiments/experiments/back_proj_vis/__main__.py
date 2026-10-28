import argparse
import logging
import pathlib
import pprint

import SimpleITK as sitk
import plotly
import plotly.express as px
import torch

from reg23_experiments.data.save_data.ct_electrodes import CTElectrodeSaveManager
from reg23_experiments.data.save_data.transformations import TransformationSaveManager
from reg23_experiments.data.save_data.xray_electrodes import XRayElectrodeSaveManager
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.io.command_line import get_string_required
from reg23_experiments.io.image import XrayDICOM, read_dicom
from reg23_experiments.io.serialize import serialize_recursive
from reg23_experiments.io.sitk import DCMSeriesInfo, find_ct_series, load_ct_series
from reg23_experiments.ops.backprojection import ordered_points_to_rays, ray_sphere_intersection
from reg23_experiments.utils import logs_setup


def acquire_ct_series_uid(ct_path: str) -> str | Error:
    series: dict[str, DCMSeriesInfo] | Error = find_ct_series(ct_path)
    if isinstance(series, Error):
        raise Exception(f"Failed to open CT from path '{ct_path}': {series.description}")
    if not series:
        return Error(f"No CT series found at path '{ct_path}'.")
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
    ct_path: str,  #
    xray_paths: list[str],  #
) -> int:
    ct_series_uid: str | Error = acquire_ct_series_uid(ct_path)
    if isinstance(ct_series_uid, Error):
        logger.error(f"Failed to acquire CT series UID from path '{ct_path}': {ct_series_uid.description}")
        return 1

    xray_images: list[XrayDICOM] = [  #
        read_dicom(xray_path)  #
        for xray_path in xray_paths  #
    ]

    ct_electrode_save_manager = CTElectrodeSaveManager(pathlib.Path("data/ct_electrode_save_data"))
    xray_electrode_save_manager = XRayElectrodeSaveManager(pathlib.Path("data/app_electrode_save_data"))
    transformation_save_manager = TransformationSaveManager(pathlib.Path("data/app_transformation_save_data"))

    ground_truth: torch.Tensor | None = ct_electrode_save_manager.get(ct_series_uid)
    if ground_truth is None:
        logger.error(f"No ground truth CT electrode positions found for {ct_series_uid}")
        return 1

    volume_image: sitk.Image | Error = load_ct_series(ct_path, ct_series_uid)
    if isinstance(volume_image, Error):
        logger.error(f"Failed to load CT volume from path '{ct_path}': {volume_image.description}")
        return 1
    volume_diag = torch.tensor(volume_image.GetSize()).flip(dims=(0,)) * torch.tensor(volume_image.GetSpacing())
    logger.info("Volume diagonal = ({:.3f}, {:.3f}, {:.3f}".format(*volume_diag.tolist()))

    # ToDo: Figure out
    ground_truth -= 0.5 * volume_diag.unsqueeze(0)

    centre = ground_truth.mean(dim=0)
    # radius = 2.0 * torch.linalg.vector_norm(ground_truth - centre.unsqueeze(0), dim=1).max()
    radius = 200.0

    figure = plotly.graph_objs.Figure()

    figure.add_scatter3d(  #
        x=ground_truth[:, 0].numpy(),  #
        y=ground_truth[:, 1].numpy(),  #
        z=ground_truth[:, 2].numpy(),  #
        mode="markers",  #
        name="ground truth",  #
    )

    for j, image in enumerate(xray_images):
        uid = image["uid"]
        source_distance = image["scene_geometry"].source_position(torch.device("cpu"))[2].item()
        tensor = xray_electrode_save_manager.get(uid)
        if tensor is None:
            logger.warning(f"No electrodes found for X-ray UID '{uid}'.")
            continue
        xray_diag = image["spacing"] * torch.tensor(image["image"].size(), dtype=torch.float64).flip(dims=(0,))
        # ToDo: Figure out
        tensor -= 0.5 * xray_diag.unsqueeze(0)

        tr: Transformation | Error = transformation_save_manager.get_transformation(  #
            source_uid=ct_series_uid,  #
            destination_uid=uid,  #
            name="gold_standard",  #
        )
        if isinstance(tr, Error):
            logger.error(f"Failed to get transformation 'gold-standard' for X-ray UID '{uid}'; {tr.description}")
            continue

        ray_points, ray_unit_dirs = ordered_points_to_rays(  #
            tensor,  #
            tr,  #
            source_distance,  #
        )
        for i, (ray_point, ray_unit_dir) in enumerate(zip(ray_points, ray_unit_dirs)):
            intersections = ray_sphere_intersection(ray_point, ray_unit_dir, centre, radius)
            if len(intersections) < 2:
                logger.warning(f"Ray {i} of X-ray UID '{uid}' didn't intersect the view sphere.")
                continue
            figure.add_scatter3d(  #
                x=intersections[:, 0].numpy(),  #
                y=intersections[:, 1].numpy(),  #
                z=intersections[:, 2].numpy(),  #
                mode="lines",  #
                line=dict(width=3, color=px.colors.qualitative.Plotly[1 + j]),  #
                name=f"{i} of {uid[-5:]}",  #
            )

    figure.update_layout(scene_aspectmode="data")  # equal axis scaling
    figure.show()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", epilog="")

    parser.add_argument("-p", "--ct-path", type=str, default=None,  #
        help="Give a path to a .nrrd file, .nii file or directory of .dcm files containing CT data to process.")
    parser.add_argument("-x", "--xray-paths", type=str, nargs='*',  #
        help="Give a path to a DICOM file containing an X-ray image to register the CT image to. If this is provided, "
             "the X-ray will by used instead of any DRR.")
    parser.add_argument("-l", "--log-level", type=str, default=None,  #
        help=f"Set the log level; options: [{", ".join(logging._nameToLevel.keys())}]; default = INFO.")

    args = parser.parse_args()

    logger = logs_setup.setup_logger(level=args.log_level)

    exit(main(  #
        ct_path=args.ct_path,  #
        xray_paths=args.xray_paths,  #
    ))
