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

__all__ = ["LoadedVolume", "read_volume", "load_volume", "load_cached_volume"]

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals(
    [sinogram.VolumeSpec, sinogram.SinogramClassic, sinogram.SinogramHEALPix, LinearRange, SceneGeometry,
     Transformation])


class LoadedVolume(NamedTuple):
    data: torch.Tensor
    spacing: torch.Tensor


def read_volume(path: pathlib.Path) -> LoadedVolume:
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
        logger.info("Loading CT DICOM data from directory '{}'...".format(str(path)))
        files = [elem for elem in path.iterdir() if elem.is_file() and elem.suffix == ".dcm"]
        if len(files) == 0:
            raise Exception("Error: no DICOM (.dcm) files found in given directory '{}'.".format(str(path)))
        logger.info("Reading {} DICOM files...".format(len(files)))
        slices = [pydicom.dcmread(f) for f in tqdm(files)]
        logger.info("Done.")
        try:
            # Sort by z-position
            slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
        except AttributeError:
            # Fallback: sort by instance number
            slices.sort(key=lambda ds: int(ds.InstanceNumber))
            logger.warning("Sorting CT volume slices by InstanceNumber, as ImagePositionPatient not available")
        pixel_spacing = slices[0]["PixelSpacing"]
        slice_spacing = (torch.tensor([slices[0]["ImagePositionPatient"][i] for i in range(3)]) - torch.tensor(
            [slices[1]["ImagePositionPatient"][i] for i in range(3)])).norm()
        spacing = torch.tensor([pixel_spacing[1],  # column spacing (x-direction)
                                pixel_spacing[0],  # row spacing (y-direction)
                                slice_spacing  # slice spacing (z-direction)
                                ])
        volume = torch.stack([torch.tensor(pydicom.pixels.pixel_array(s)) for s in slices])
        logger.info("{}".format(slices[0]["ImageOrientationPatient"]))
        return LoadedVolume(volume, spacing)
    raise Exception("Given path '{}' is not a file or directory.".format(str(path)))


def load_volume(path: pathlib.Path, *, hu_cutoff: float = -800.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load a CT volume from a file or directory

    :param path: Path to a .nrrd file (a CT volume) or a directory containing multiple .dcm files (one for each slice
    of a CT volume)
    :param hu_cutoff: The cutoff lower bound in Hounsfield units to clamp to.
    :type path: pathlib.Path
    :return: (The ct volume, a tensor of size (3,): the spacing of the volume voxels)
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    loaded_volume = read_volume(path)
    sizes = loaded_volume.data.size()
    spacing = loaded_volume.spacing
    logger.info("CT data volume size and spacing = [{} x {} x {}]; [{} x {} x {}] mm"
                "".format(sizes[0], sizes[1], sizes[2], spacing[0], spacing[1], spacing[2]))
    image = loaded_volume.data.to(dtype=torch.float32)
    image[image < hu_cutoff] = hu_cutoff
    image -= image.min()
    image /= image.max()

    return image, spacing


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
