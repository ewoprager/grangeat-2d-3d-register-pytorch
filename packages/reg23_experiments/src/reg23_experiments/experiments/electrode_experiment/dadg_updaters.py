import logging
import pathlib
from typing import Any

import torch
import zarr

from reg23_experiments.data.segmentation import NamedPoints3D
from reg23_experiments.data.structs import Error, Transformation
from reg23_experiments.io.helpers import contents_based_hash_slices
from reg23_experiments.io.large import large_volume_to_zarr
from reg23_experiments.ops.data_manager import dadg_updater

__all__ = ["get_micro_ct_uid", "get_micro_ct_array", "transform_micro_fiducials"]

logger = logging.getLogger(__name__)


@dadg_updater(names_returned=["micro_ct_uid"])
def get_micro_ct_uid(  #
    *,  #
    micro_ct_slice_path: str | pathlib.Path | None,  #
) -> dict[str, Any] | Error:
    if micro_ct_slice_path is None:
        return {"micro_ct_uid": None}
    file = pathlib.Path(micro_ct_slice_path)
    if not file.is_file():
        return Error(f"Chosen microCT file '{str(file)}' doesn't exist.")
    logger.info(f"Retrieving hash of microCT volume contents...")
    uid: str | Error = contents_based_hash_slices(file)
    if isinstance(uid, Error):
        logger.error(f"\tFailed: {uid.description}")
        return uid
    logger.info(f"\tDone; UID = '{uid}'")
    return {"micro_ct_uid": uid}


@dadg_updater(names_returned=["micro_ct_array", "micro_ct_spacing"])
def get_micro_ct_array(  #
    *,  #
    micro_ct_slice_path: str | pathlib.Path | None,  #
    micro_ct_uid: str | None,  #
    micro_ct_zarr_path: str | pathlib.Path,  #
) -> dict[str, Any] | Error:
    if micro_ct_uid is None:
        return {  #
            "micro_ct_array": None,  #
            "micro_ct_spacing": None,  #
        }
    # micro_ct_slice_path should not be None if micro_ct_uid is not None; see `get_micro_ct_uid`
    assert micro_ct_slice_path is not None

    group = zarr.open_group(micro_ct_zarr_path)
    if micro_ct_uid in group.array_keys():
        logger.info(f"Get microCT by UID: Array already present in zarr cache")
        array = group[micro_ct_uid]
    else:
        logger.info(f"Get microCT by UID: Array not present in zarr cache; streaming in...")
        array: zarr.Array | Error = large_volume_to_zarr(  #
            group=group,  #
            volume_id=micro_ct_uid,  #
            a_slice_file_path=pathlib.Path(micro_ct_slice_path),  #
        )
        if isinstance(array, Error):
            logger.error(f"\tFailed: {array.description}")
            return array
        logger.info(f"\tDone; zarr cache entry created for volume '{micro_ct_uid}'")
    return {  #
        "micro_ct_array": array,  #
        "micro_ct_spacing": torch.tensor(array.attrs["spacing"], dtype=torch.float64),  #
    }


@dadg_updater(names_returned=["transformed_micro_fiducials"])
def transform_micro_fiducials(  #
    *,  #
    micro_ct_transformation: Transformation,  #
    micro_fiducial_points: NamedPoints3D,  #
) -> dict[str, Any]:
    device = torch.device("cpu")
    transformation = micro_ct_transformation.to(device=device)
    input_vectors = micro_fiducial_points.data.cpu()
    inputs_homogeneous = torch.cat((  #
        input_vectors,  #
        torch.ones(input_vectors.size()[:-1], dtype=torch.float64, device=device).unsqueeze(-1),  #
    ), dim=-1)
    outputs_homogeneous = torch.einsum("ji,...i->...j", transformation.get_h(device), inputs_homogeneous)
    output_points = outputs_homogeneous[..., 0:3]
    return {"transformed_micro_fiducials": NamedPoints3D(names=micro_fiducial_points.names, data=output_points)}
