import logging
import pathlib
import hashlib
from typing import NamedTuple, Type, TypeVar

logger = logging.getLogger(__name__)

import torch
import nrrd
import pydicom

from registration.lib.structs import *
from registration.lib import sinogram

torch.serialization.add_safe_globals(
    [sinogram.VolumeSpec, sinogram.DrrSpec, sinogram.SinogramClassic, sinogram.SinogramHEALPix, LinearRange,
        SceneGeometry, Transformation])


class LoadedVolume(NamedTuple):
    data: torch.Tensor
    spacing: torch.Tensor


def read_volume(path: pathlib.Path) -> LoadedVolume:
    if path.is_file():
        logger.info("Loading CT data file '{}'...".format(str(path)))
        if path.suffix == ".nrrd":
            data, header = nrrd.read(str(path))
        else:
            raise Exception("Error: file {} is of unrecognised type.".format(str(path)))
        logger.info("CT data file loaded.")
        directions = torch.tensor(header['space directions'])
        spacing = directions.norm(dim=1).flip(dims=(0,))
        return LoadedVolume(torch.tensor(data), spacing)
    if path.is_dir():
        logger.info("Loading CT DICOM data from directory '{}'...".format(str(path)))
        files = [elem for elem in path.iterdir() if elem.is_file() and elem.suffix == ".dcm"]
        if len(files) == 0:
            raise Exception("Error: no DICOM (.dcm) files found in given directory '{}'.".format(str(path)))
        slices = [pydicom.dcmread(f) for f in files]
        try:
            # Sort by z-position
            slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
        except AttributeError:
            # Fallback: sort by instance number
            slices.sort(key=lambda ds: int(ds.InstanceNumber))
        pixel_spacing = slices[0]["PixelSpacing"]
        slice_spacing = (torch.tensor([slices[0]["ImagePositionPatient"][i] for i in range(3)]) - torch.tensor(
            [slices[1]["ImagePositionPatient"][i] for i in range(3)])).norm()
        spacing = torch.tensor([pixel_spacing[1],  # column spacing (x-direction)
            pixel_spacing[0],  # row spacing (y-direction)
            slice_spacing  # slice spacing (z-direction)
        ])
        volume = torch.stack([torch.tensor(pydicom.pixels.pixel_array(s)) for s in slices])
        return LoadedVolume(volume, spacing)
    raise Exception("Given path '{}' is not a file or directory.".format(str(path)))


def deterministic_hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


def deterministic_hash_int(x: int) -> str:
    return hashlib.sha256(x.to_bytes(64)).hexdigest()


def deterministic_hash_type(tp: type) -> str:
    string = "{}.{}".format(tp.__module__, tp.__qualname__)
    return deterministic_hash_string(string)


def deterministic_hash_combo(*hex_digests: str) -> str:
    combined = b''.join(bytes.fromhex(h) for h in hex_digests)
    return hashlib.sha256(combined).hexdigest()


SinogramType = TypeVar('SinogramType')


def deterministic_hash_sinogram(path: str, sinogram_type: Type[SinogramType], sinogram_size: int,
                                downsample_factor: int) -> str:
    return deterministic_hash_combo(deterministic_hash_string(path), deterministic_hash_type(sinogram_type),
        deterministic_hash_int(sinogram_size), deterministic_hash_int(downsample_factor))


def load_volume(path: pathlib.Path, *, downsample_factor=1) -> Tuple[torch.Tensor, torch.Tensor]:
    loaded_volume = read_volume(path)
    logger.info("Processing CT data...")
    sizes = loaded_volume.data.size()
    logger.info("CT data volume size = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    image = loaded_volume.data.to(dtype=torch.float32)
    image[image < -800.0] = -800.0
    image -= image.min()
    image /= image.max()
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool3d(downsample_factor)
        image = down_sampler(image.unsqueeze(0))[0]
        sizes = image.size()
        logger.info("CT volume size after down-sampling = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    spacing = float(downsample_factor) * loaded_volume.spacing
    logger.info("CT voxel spacing after down-sampling = [{} x {} x {}] mm".format(spacing[0], spacing[1], spacing[2]))
    logger.info("CT data file processed.")
    return image, spacing


def read_dicom(path: str, *, downsample_factor=1):
    logger.info("Loading X-ray DICOM file {}...".format(path))
    dataset = pydicom.dcmread(path)
    logger.info("DICOM file loaded.")
    logger.info("Processing DICOM data...")
    image = torch.tensor(pydicom.pixels.pixel_array(dataset), dtype=torch.float32)
    logger.info("X-ray image size = [{} x {}]".format(image.size()[0], image.size()[1]))
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool2d(downsample_factor)
        image = down_sampler(image.unsqueeze(0))[0]
        logger.info("X-ray image size after down-sampling = [{} x {}]".format(image.size()[0], image.size()[1]))
    if "PixelSpacing" in dataset:
        spacing = float(downsample_factor) * torch.tensor([dataset["PixelSpacing"][1],  # column spacing (x-direction)
            dataset["PixelSpacing"][0]  # row spacing (y-direction)
        ])
        logger.info("X-ray pixel spacing = [{} x {}] mm".format(spacing[0], spacing[1]))
        scene_geometry = SceneGeometry(dataset["DistanceSourceToPatient"].value)
        logger.info("X-ray distance source-to-patient = {} mm".format(scene_geometry.source_distance))
    else:
        spacing = float(downsample_factor) * torch.tensor(
            [dataset["ImagerPixelSpacing"][1],  # column spacing (x-direction)
                dataset["ImagerPixelSpacing"][0]  # row spacing (y-direction)
            ])
        logger.info("X-ray imager pixel spacing = [{} x {}] mm".format(spacing[0], spacing[1]))
        scene_geometry = SceneGeometry(dataset["DistanceSourceToDetector"].value)
        logger.info("X-ray distance source-to-detector = {} mm".format(scene_geometry.source_distance))

    logger.info("X-ray DICOM file processed.")
    return image, spacing, scene_geometry


def load_cached_volume(cache_directory: str, sinogram_hash: str):
    file: str = cache_directory + "/volume_spec_{}.pt".format(sinogram_hash)
    try:
        volume_spec = torch.load(file)
    except FileNotFoundError:
        logger.warning("No cache file '{}' found.".format(file))
        return None
    volume_downsample_factor = volume_spec.downsample_factor
    sinogram3d = volume_spec.sinogram
    logger.info(
        "Loaded cached volume spec from '{}'; sinogram size = [{} x {} x {}]".format(file, sinogram3d.data.size()[0],
            sinogram3d.data.size()[1], sinogram3d.data.size()[2]))
    return volume_downsample_factor, sinogram3d


def load_cached_drr(cache_directory: str, ct_volume_path: str):
    file: str = cache_directory + "/drr_spec_{}.pt".format(deterministic_hash_string(ct_volume_path))
    try:
        drr_spec = torch.load(file)
    except FileNotFoundError:
        logger.warning("No cache file '{}' found.".format(file))
        return None
    assert drr_spec.ct_volume_path == ct_volume_path
    detector_spacing = drr_spec.detector_spacing
    scene_geometry = drr_spec.scene_geometry
    drr_image = drr_spec.image
    transformation_ground_truth = drr_spec.transformation
    logger.info("Loaded cached drr spec from '{}'".format(file))
    return detector_spacing, scene_geometry, drr_image, transformation_ground_truth
