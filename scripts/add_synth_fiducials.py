import argparse
import pathlib
import nrrd
from typing import Callable

import torch

from reg23_experiments.utils import logs_setup
from reg23_experiments.data.structs import Error
from reg23_experiments.io import stradview, volume

type MarkerFunction = Callable[[torch.Tensor], torch.Tensor]


def marker_function(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Maps positions given in 3D Cartesian coordinates at mm scale to intensities
    :param coordinates: tensor of size (..., 3)
    :return: tensor of size (...)
    """
    value = 30000.0
    radius = 2.5
    dist = torch.linalg.vector_norm(coordinates, dim=-1)
    return value * torch.exp(-(dist / radius).square())


def construct_marker_image(*, size: torch.Size, spacing: torch.Tensor, origin: torch.Tensor,
                           markers: list[tuple[torch.Tensor, MarkerFunction]]) -> torch.Tensor:
    """
    Constructs a tensor of the given size as the sum of the given markers.
    :param size: (z, y, x)
    :param spacing: (x, y, z)
    :param origin: (x, y, z) position of the (0, 0, 0) voxel relative to the origin
    :param markers: a list of markers parametrised by (position, intensity function)
    :return:
    """
    assert len(size) == 3
    assert spacing.size() == torch.Size([3])
    assert origin.size() == torch.Size([3])

    voxel_x_positions = origin[0] + torch.arange(size[2]) * spacing[0]
    voxel_y_positions = origin[1] + torch.arange(size[1]) * spacing[1]
    voxel_z_positions = origin[2] + torch.arange(size[0]) * spacing[2]
    voxel_z_positions, voxel_y_positions, voxel_x_positions = torch.meshgrid(voxel_z_positions, voxel_y_positions,
                                                                             voxel_x_positions)
    voxel_positions = torch.stack((voxel_x_positions, voxel_y_positions, voxel_z_positions), dim=-1)

    ret = torch.zeros(size=size)
    for m in markers:
        if m[0].size() != torch.Size([3]):
            logger.error(f"Marker positions should be 3 dimensional; got '{m[0]}'.")
            continue
        ret += m[1](voxel_positions - m[0])
    return ret


def main(*, ct_path: pathlib.Path, landmark_path: pathlib.Path, output_path: pathlib.Path | None):
    assert ct_path.is_dir()
    series = volume.find_dicom_series_in_directory(ct_path, check_for_dcm_suffix=False)
    logger.info(series)
    ct_volume, ct_spacing, image_position_patient = volume.read_dicom_series_from_directory(ct_path, series_number=
    max(series, key=lambda t: t[1])[0], check_for_dcm_suffix=False)

    logger.info(
        "CT volume loaded, size=[{} x {} x {}]; spacing=({:.3f}, {:.3f}, {:.3f}); image position patient=({:.3f}, "
        "{:.3f}, {:.3f})".format(  #
            ct_volume.size()[0], ct_volume.size()[1], ct_volume.size()[2],  #
            ct_spacing[0].item(), ct_spacing[1].item(), ct_spacing[2].item(),  #
            image_position_patient[0].item(), image_position_patient[1].item(), image_position_patient[2].item(),  #
        ))

    landmarks: list[tuple[str, torch.Tensor]] | Error = stradview.extract_landmarks(landmark_path)
    if isinstance(landmarks, Error):
        logger.error(f"Failed to extract landmarks: {landmarks.description}")
        exit(1)

    difference_volume = construct_marker_image(  #
        size=ct_volume.size(), spacing=ct_spacing, origin=image_position_patient, markers=[  #
            (position, marker_function)  #
            for _, position in landmarks  #
        ])

    output_image = ct_volume + difference_volume

    if output_path is None:
        output_path = ct_path.parent / f"{ct_path.stem}_synth_fid.nrrd"

    header = {  #
        "space": "left-posterior-superior",  #
        "dimension": 3,  #
        "space directions": [  #
            [0.0, 0.0, ct_spacing[2].item()],  # Z axis
            [0.0, ct_spacing[1].item(), 0.0],  # Y axis
            [ct_spacing[0].item(), 0.0, 0.0],  # X axis
        ],  #
        "space origin": image_position_patient.tolist(),  #
        "encoding": "raw"  #
    }
    nrrd.write(str(output_path), output_image.cpu().contiguous().numpy(), header=header)
    logger.info(f"Output saved to '{str(output_path)}'.")


if __name__ == "__main__":
    logger = logs_setup.setup_logger()

    _parser = argparse.ArgumentParser(description="", epilog="")
    _parser.add_argument("-c", "--ct-path", type=str, default=None, required=True,
                         help="Give a path to a .nrrd file, .nii file or directory of DICOM files containing CT data to"
                              " modify.")
    _parser.add_argument("-l", "--landmark-path", type=str, default=None, required=True,
                         help="Give a path to a CSV file containing landmark data exported from StradView.")
    _parser.add_argument("-o", "--output-path", type=str, default=None, required=False,
                         help="Give a path at which to save the modified CT data.")
    _args = _parser.parse_args()

    main(ct_path=pathlib.Path(_args.ct_path), landmark_path=pathlib.Path(_args.landmark_path),
         output_path=None if _args.output_path is None else pathlib.Path(_args.output_path))
