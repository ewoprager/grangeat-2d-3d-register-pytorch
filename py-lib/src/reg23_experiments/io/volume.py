import logging
import pathlib
from typing import NamedTuple, Tuple, Literal, Union

import torch
import nrrd
import pydicom
import nibabel
from tqdm import tqdm

from reg23_experiments.data.structs import SceneGeometry, Transformation, LinearRange
from reg23_experiments.data import sinogram
from reg23_experiments.ops.volume import fit_line_3d, point_line_distance_3d

__all__ = ["LoadedVolume", "read_volume", "load_volume", "load_cached_volume"]

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals(
    [sinogram.VolumeSpec, sinogram.SinogramClassic, sinogram.SinogramHEALPix, LinearRange, SceneGeometry,
     Transformation])


class LoadedVolume(NamedTuple):
    data: torch.Tensor
    spacing: torch.Tensor


def read_volume(path: pathlib.Path, *, check_for_dcm_suffix_if_dir: bool = True) -> LoadedVolume:
    if path.is_file():
        logger.info("Loading CT data file '{}'...".format(str(path)))
        # Obtain a data tensor and a tensor of voxel spacing
        if path.suffix == ".nrrd":
            data, header = nrrd.read(str(path))
            data = torch.tensor(data)
            directions = torch.tensor(header['space directions'], dtype=torch.float32)
            spacing = directions.norm(dim=1).flip(dims=(0,))
            logger.warning("Don't know how to read ImageOrientationPatient from .nrrd files.")
        elif path.suffix == ".nii":
            img = nibabel.load(str(path))
            data = torch.tensor(img.get_fdata())
            data = data.permute(*reversed(range(data.ndim)))
            spacing = torch.tensor(img.header.get_zooms(), dtype=torch.float32)[0:3]
            if path.stem == "PhantomCT":
                # ToDo: Double check this:
                logger.warning("Applying super hacky adjustment; check it's correct!")
                data -= 1000.0  # super hacky adjustment for one particular file which appears to be HU + 1000
            logger.warning("Don't know how to read ImageOrientationPatient from .nii files.")
        else:
            raise Exception("Error: file {} is of unrecognised type.".format(str(path)))
        # Make sure there aren't unnecessary dimensions
        if len(data.size()) > 3:
            data = data.squeeze()
        if len(data.size()) != 3:
            raise Exception("Error: CT volume file '{}' contains invalid size '{}'"
                            "".format(str(path), str(data.size())))
        return LoadedVolume(data, spacing)
    if path.is_dir():
        logger.info(f"Loading CT DICOM data from directory '{str(path)}'")
        files = [elem for elem in path.iterdir() if elem.is_file()]
        if check_for_dcm_suffix_if_dir:
            files = [elem for elem in files if elem.suffix == ".dcm"]
        if len(files) == 0:
            raise Exception("Error: no {}files found in given directory '{}'.".format(
                "DICOM (.dcm) " if check_for_dcm_suffix_if_dir else "", str(path)))
        slices = [pydicom.dcmread(f) for f in tqdm(files, desc=f"Reading DICOM files")]
        # Filter so we only have slices for which there is "PixelData"
        slices = [s for s in slices if "PixelData" in s]
        # Filter again for slices for which we have "ImagePositionPatient". We will revert back to the original slices
        # if the positioned slices are not evenly spaced.
        positioned_slices = [s for s in slices if "ImagePositionPatient" in s]
        # Extract the positions from the slices
        slice_positions = torch.tensor([[s["ImagePositionPatient"][i] for i in range(3)] for s in positioned_slices])
        # Fit a line to the slice position point cloud and remove outliers
        line_point, line_direction = fit_line_3d(slice_positions)
        point_line_distances = point_line_distance_3d(points=slice_positions, line_point=line_point,
                                                      line_direction=line_direction)
        median_distance = point_line_distances.median()
        mad = (point_line_distances - median_distance).abs().median()
        threshold_distance = median_distance + 3.0 * mad
        outlier_indices = torch.nonzero(point_line_distances > threshold_distance).squeeze(1)
        if len(outlier_indices) > 0:
            logger.warning(f"Removing {len(outlier_indices)} slices as 'ImagePositionPatient' values were outliers.")
            positioned_slices = [positioned_slices[i] for i in range(len(positioned_slices)) if
                                 i not in outlier_indices]

        # Sort slices by distance along the fitted line
        def distance_of_slice(_slice) -> float:
            position = torch.tensor([_slice["ImagePositionPatient"][i] for i in range(3)])
            return torch.dot(position, line_direction).item()

        positioned_slices.sort(key=distance_of_slice)
        # Extract the spacings between each slice
        slice_positions = torch.tensor([[s["ImagePositionPatient"][i] for i in range(3)] for s in positioned_slices])
        slice_spacings = torch.linalg.vector_norm(slice_positions[1:, :] - slice_positions[:-1, :], dim=-1)
        # Check for spacings of zero, and remove those slices
        zero_spacings = torch.isclose(slice_spacings, torch.zeros(1), atol=1.0e-3)
        zero_spacing_indices = zero_spacings.nonzero().squeeze(1)
        if len(zero_spacing_indices) > 0:
            logger.warning(f"Removing {len(zero_spacing_indices)} slices, as they have zero spacing.")
            positioned_slices = [positioned_slices[i] for i in range(len(positioned_slices)) if
                                 i not in zero_spacing_indices]
            slice_positions = torch.tensor(
                [[s["ImagePositionPatient"][i] for i in range(3)] for s in positioned_slices])
            slice_spacings = torch.linalg.vector_norm(slice_positions[1:, :] - slice_positions[:-1, :], dim=-1)
        # Set the slice spacing for the volume as a whole to the median spacing, and look for inconsistencies
        slice_spacing = slice_spacings.median()
        bad_spacings = ~torch.isclose(slice_spacings, slice_spacing, atol=1.0e-3)
        bad_spacing_indices = bad_spacings.nonzero().squeeze(1)
        if len(bad_spacing_indices) > 0:
            logger.warning(
                "Median slice spacing is {:.4f}; some slices deviate from this:".format(slice_spacing.item()))
            for index in bad_spacing_indices:
                logger.warning(
                    "Spacing between slices {} and {} is {:.4f}".format(index, index + 1, slice_spacings[index].item()))
            logger.warning(
                "Slice positioning was inconsistent, so reverting to using all slices (regardless of whether "
                "'ImagePositionPatient' is provided, and sorting instead by 'InstanceNumber'.")
            slices.sort(key=lambda ds: int(ds.InstanceNumber))
        else:
            logger.info("No inconsistencies in slice spacing, so sticking to using only slices that have "
                        "'ImagePositionPatient'.")
            slices = positioned_slices

        # Extract the in-plane spacing and assemble the spacing vector
        pixel_spacing = slices[0]["PixelSpacing"]
        spacing = torch.tensor([pixel_spacing[1],  # column spacing (x-direction)
                                pixel_spacing[0],  # row spacing (y-direction)
                                slice_spacing  # slice spacing (z-direction)
                                ])
        volume = torch.stack([torch.tensor(pydicom.pixels.pixel_array(s)) for s in slices])
        logger.info("{}".format(slices[0]["ImageOrientationPatient"]))
        return LoadedVolume(volume, spacing)
    raise Exception("Given path '{}' is not a file or directory.".format(str(path)))


def load_volume(path: pathlib.Path, *, hu_cutoff: float = -800.0, mu_water: float = 0.02,
                check_for_dcm_suffix_if_dir: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a CT volume from a file or directory

    :param path: Path to a .nrrd file (a CT volume) or a directory containing multiple .dcm files (one for each slice
    of a CT volume)
    :param hu_cutoff: The cutoff lower bound in Hounsfield units to clamp to.
    :param mu_water: The attenuation coefficient of water in [mm^-1]
    :return: (The ct volume, a tensor of size (3,): the spacing of the volume voxels)
    :rtype: tuple[torch.Tensor, torch.Tensor], (the CT volume of attenuation coefficient, voxel spacing in [mm])
    """
    loaded_volume = read_volume(path, check_for_dcm_suffix_if_dir=check_for_dcm_suffix_if_dir)
    sizes = loaded_volume.data.size()
    spacing = loaded_volume.spacing
    logger.info("CT data volume size and spacing = [{} x {} x {}]; [{} x {} x {}] mm"
                "".format(sizes[0], sizes[1], sizes[2], spacing[0], spacing[1], spacing[2]))
    image_hu = loaded_volume.data.to(dtype=torch.float32)
    image_hu[image_hu < hu_cutoff] = hu_cutoff
    image_mu = mu_water * (1.0 + image_hu / 1000.0)

    return image_mu, spacing


def load_cached_volume(cache_directory: str, sinogram_hash: str) -> Tuple[int, sinogram.Sinogram] | None:
    file: str = cache_directory + "/volume_spec_{}.pt".format(sinogram_hash)
    try:
        volume_spec = torch.load(file)
    except FileNotFoundError:
        logger.warning("No cache file '{}' found.".format(file))
        return None
    volume_downsample_factor = volume_spec.downsample_factor
    sinogram3d = volume_spec.sinogram
    logger.info("Loaded cached volume spec from '{}'; sinogram size = [{} x {} x {}]"
                "".format(file, sinogram3d.data.size()[0], sinogram3d.data.size()[1], sinogram3d.data.size()[2]))
    return volume_downsample_factor, sinogram3d
