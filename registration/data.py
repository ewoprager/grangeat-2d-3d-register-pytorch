import logging

logger = logging.getLogger(__name__)

import torch
import nrrd
import pydicom

from registration.lib.structs import *
from registration.lib.sinogram import *


def deterministic_hash(text: str):
    ret = 0
    for ch in text:
        ret = (ret * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
    return ret


def read_nrrd(path: str, downsample_factor=1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logger.info("Loading CT data file {}...".format(path))
    data, header = nrrd.read(path)
    logger.info("CT data file loaded.")
    logger.info("Processing CT data...")
    sizes = header['sizes']
    logger.info("CT data volume size = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    data = torch.tensor(data, device="cpu")
    image = torch.maximum(data.type(torch.float32) + 1000., torch.tensor([0.], device=data.device))
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool3d(downsample_factor)
        image = down_sampler(image[None, :, :, :])[0]
    sizes = image.size()
    logger.info("CT data volume size after down-sampling = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    bounds = torch.Tensor([image.min().item(), image.max().item()])
    logger.info("CT data value range = ({:.3f}, {:.3f})".format(bounds[0], bounds[1]))
    bounds[1] *= 10000.
    directions = torch.tensor(header['space directions'])
    spacing = float(downsample_factor) * directions.norm(dim=1).flip(dims=(0,))
    logger.info("CT voxel spacing = [{} x {} x {}] mm".format(spacing[0], spacing[1], spacing[2]))
    logger.info("CT data file processed.")
    return image, spacing, bounds


def read_dicom(path: str):
    logger.info("Loading X-ray DICOM file {}...".format(path))
    dataset = pydicom.dcmread(path)
    logger.info("DICOM file loaded.")
    logger.info("Processing DICOM data...")
    image = torch.tensor(pydicom.pixels.pixel_array(dataset), dtype=torch.float32)
    logger.info("X-ray image size = [{} x {}]".format(image.size()[0], image.size()[1]))
    spacing = dataset["PixelSpacing"]
    spacing = torch.tensor([spacing[0], spacing[1]])
    logger.info("X-ray pixel spacing = [{} x {}] mm".format(spacing[0], spacing[1]))
    scene_geometry = SceneGeometry(dataset["DistanceSourceToPatient"].value)
    logger.info("X-ray distance source-to-patient = {} mm".format(scene_geometry.source_distance))
    logger.info("X-ray DICOM file processed.")
    return image, spacing, scene_geometry



def load_cached_volume(cache_directory: str, ct_volume_path: str):
    file: str = cache_directory + "/volume_spec_{}.pt".format(deterministic_hash(ct_volume_path))
    try:
        volume_spec = torch.load(file)
    except:
        logger.warning("No cache file '{}' found.".format(file))
        return None
    if not isinstance(volume_spec, VolumeSpec):
        logger.error("Cache file '{}' invalid.".format(file))
        return None
    assert ct_volume_path == volume_spec.ct_volume_path
    volume_downsample_factor = volume_spec.downsample_factor
    sinogram3d = volume_spec.sinogram
    logger.info(
        "Loaded cached volume spec from '{}'; sinogram size = [{} x {} x {}]".format(file, sinogram3d.data.size()[0],
                                                                                     sinogram3d.data.size()[0],
                                                                                     sinogram3d.data.size()[0]))
    return volume_downsample_factor, sinogram3d


def load_cached_volume_fibonacci(cache_directory: str, ct_volume_path: str):
    file: str = cache_directory + "/volume_spec_fibonacci_{}.pt".format(deterministic_hash(ct_volume_path))
    try:
        volume_spec = torch.load(file)
    except:
        logger.warning("No cache file '{}' found.".format(file))
        return None
    if not isinstance(volume_spec, VolumeSpecFibonacci):
        logger.error("Cache file '{}' invalid.".format(file))
        return None
    assert ct_volume_path == volume_spec.ct_volume_path
    volume_downsample_factor = volume_spec.downsample_factor
    sinogram3d = volume_spec.sinogram
    logger.info("Loaded cached Fibonacci volume spec from '{}'".format(file))
    return volume_downsample_factor, sinogram3d


def load_cached_drr(cache_directory: str, ct_volume_path: str):
    file: str = cache_directory + "/drr_spec_{}.pt".format(deterministic_hash(ct_volume_path))
    try:
        drr_spec = torch.load(file)
    except:
        logger.warning("No cache file '{}' found.".format(file))
        return None
    if not isinstance(drr_spec, DrrSpec):
        logger.error("Cache file '{}' invalid.".format(file))
        return None
    assert drr_spec.ct_volume_path == ct_volume_path
    detector_spacing = drr_spec.detector_spacing
    scene_geometry = drr_spec.scene_geometry
    drr_image = drr_spec.image
    fixed_image = drr_spec.sinogram
    sinogram2d_range = drr_spec.sinogram_range
    transformation_ground_truth = drr_spec.transformation
    logger.info("Loaded cached drr spec from '{}'".format(file))
    return detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation_ground_truth
