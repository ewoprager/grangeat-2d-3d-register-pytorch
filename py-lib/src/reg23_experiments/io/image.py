import logging
from typing import Tuple

import torch
import pydicom

from reg23_experiments.data.structs import SceneGeometry, LinearRange, Transformation
from reg23_experiments.data import sinogram
from reg23_experiments.io.helpers import deterministic_hash_string

__all__ = ["read_dicom", "load_cached_drr"]

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals([sinogram.DrrSpec, LinearRange, SceneGeometry, Transformation])


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


def read_dicom(path: str) -> Tuple[torch.Tensor, torch.Tensor, SceneGeometry]:
    logger.info("Loading X-ray DICOM file {}...".format(path))
    dataset = pydicom.dcmread(path)
    image = torch.tensor(pydicom.pixels.pixel_array(dataset), dtype=torch.float32)
    logger.info("X-ray image size = [{} x {}]".format(image.size()[0], image.size()[1]))

    # if "PixelIntensityRelationship" in dataset:
    #     logger.info("X-ray pixel intensity relationship = '{}'.".format(dataset["PixelIntensityRelationship"]))
    # else:
    #     logger.info("No pixel intensity relationship available for X-ray.")
    #
    # if "AcquisitionDeviceProcessingDescription" in dataset:
    #     logger.info("X-ray AcquisitionDeviceProcessingDescription = '{}'.".format(dataset[
    #     "AcquisitionDeviceProcessingDescription"]))
    # else:
    #     logger.info("No AcquisitionDeviceProcessingDescription available for X-ray.")
    #
    # if "ModalityLUTSequence" in dataset:
    #     logger.info("X-ray ModalityLUTSequence = '{}'.".format(dataset["ModalityLUTSequence"]))
    # else:
    #     logger.info("No ModalityLUTSequence available for X-ray.")
    #
    # if "ModalityLUTType" in dataset:
    #     logger.info("X-ray ModalityLUTType = '{}'.".format(dataset["ModalityLUTType"]))
    # else:
    #     logger.info("No ModalityLUTType available for X-ray.")
    #
    # if "VOILUTSequence" in dataset:
    #     logger.info("X-ray VOILUTSequence = '{}'.".format(dataset["VOILUTSequence"]))
    # else:
    #     logger.info("No VOILUTSequence available for X-ray.")
    #
    # if "VOILUTType" in dataset:
    #     logger.info("X-ray VOILUTType = '{}'.".format(dataset["VOILUTType"]))
    # else:
    #     logger.info("No VOILUTType available for X-ray.")
    #
    # if "RescaleIntercept" in dataset:
    #     logger.info("X-ray RescaleIntercept = '{}'.".format(dataset["RescaleIntercept"]))
    # else:
    #     logger.info("No RescaleIntercept available for X-ray.")

    if "DistanceSourceToPatient" in dataset and "DistanceSourceToDetector" in dataset:
        spacing_spread_ratio = float(
            dataset["DistanceSourceToDetector"].value / dataset["DistanceSourceToPatient"].value)
    else:
        spacing_spread_ratio = 1.0
        logger.warning("'DistanceSourceToPatient' and 'DistanceSourceToDetector' not both available, assuming spacing "
                       "spread ratio of {} from CT volume to detector array.".format(spacing_spread_ratio))

    if "PixelSpacing" in dataset:
        spacing = torch.tensor([dataset["PixelSpacing"][1],  # column spacing (x-direction)
                                dataset["PixelSpacing"][0]  # row spacing (y-direction)
                                ])
        logger.info("X-ray pixel spacing = [{} x {}] mm".format(spacing[0], spacing[1]))
        scene_geometry = SceneGeometry(source_distance=dataset["DistanceSourceToPatient"].value)
        logger.info("X-ray distance source-to-patient = {} mm".format(scene_geometry.source_distance))
    else:
        spacing = torch.tensor([dataset["ImagerPixelSpacing"][1],  # column spacing (x-direction)
                                dataset["ImagerPixelSpacing"][0]  # row spacing (y-direction)
                                ])
        logger.info("X-ray imager pixel spacing = [{} x {}] mm".format(spacing[0], spacing[1]))
        scene_geometry = SceneGeometry(source_distance=dataset["DistanceSourceToDetector"].value)
        logger.info("X-ray distance source-to-detector = {} mm".format(scene_geometry.source_distance))

    return image, spacing, scene_geometry
