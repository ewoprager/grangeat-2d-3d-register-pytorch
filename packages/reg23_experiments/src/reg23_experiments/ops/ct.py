import logging

import SimpleITK as sitk
import torch

from reg23_experiments.data.structs import Error
from reg23_experiments.io.volume import Volume

__all__ = ["convert_ct_to_mu", "convert_ct_to_mu_sitk"]

logger = logging.getLogger(__name__)


def convert_ct_to_mu(volume: Volume, *, dtype: torch.dtype = torch.float32, hu_cutoff: float = -1000.0,
                     mu_water: float = 0.02) -> torch.Tensor | Error:
    if volume.rescale_type is None:
        logger.warning(f"No rescale type specified for volume; assuming Hounsfield Units.")
    elif volume.rescale_type != "HU":
        return Error(f"Unsupported rescale type for CT volume: {volume.rescale_type}")
    volume_hu = volume.raw_data.to(dtype=dtype) * volume.rescale_slope + volume.rescale_intercept
    volume_hu[volume_hu < hu_cutoff] = hu_cutoff
    volume_mu = mu_water * (1.0 + volume_hu / 1000.0)
    return volume_mu


def convert_ct_to_mu_sitk(volume: sitk.Image, *, dtype: torch.dtype = torch.float32, hu_cutoff: float = -1000.0,
                          mu_water: float = 0.02) -> torch.Tensor | Error:
    try:
        array = sitk.GetArrayFromImage(volume)
    except Exception as e:
        return Error(f"Failed to get data from image: {str(e)}.")
    volume_hu = torch.tensor(array, dtype=dtype)
    volume_hu[volume_hu < hu_cutoff] = hu_cutoff
    volume_mu = mu_water * (1.0 + volume_hu / 1000.0)
    return volume_mu
