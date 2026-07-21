import logging
import pathlib

import pydicom
import torch
import traitlets
from pydicom.errors import InvalidDicomError
from tqdm import tqdm
from traitlets import TraitError

from reg23_experiments.data.structs import Error

from ._data import OneSeries, SeriesDescription, Volume
from ._loader_base import VolumeLoader

__all__ = ["DICOMVolumeLoader"]

logger = logging.getLogger(__name__)


def _extract_groupings(points: torch.Tensor, *, max_cluster_radius: float) -> list[torch.Tensor]:
    """
    :param points: (N, D) tensor of N D-dimensional points that are clustered into an unknown number of dense clusters
    :return: a list of 1D tensors containing indices of points in the points tensor
    """
    cluster_list: list[list[int]] = []
    for point_index in range(points.size()[0]):
        cluster_found: int | None = None
        for cluster_index, cluster in enumerate(cluster_list):
            cluster_avg = torch.tensor([points[index].tolist() for index in cluster]).mean(dim=0)  # size = (3,)
            distance = torch.linalg.vector_norm(points[point_index] - cluster_avg)
            if distance < max_cluster_radius:
                cluster_found = cluster_index
        if cluster_found is None:
            cluster_list.append([point_index])
        else:
            cluster_list[cluster_found].append(point_index)
    return [torch.tensor(cluster) for cluster in cluster_list]


class _SeriesMetadata(traitlets.HasTraits):
    uid: str | None = traitlets.Unicode(allow_none=True)
    number: int | None = traitlets.Integer(allow_none=True)
    description: str | None = traitlets.Unicode(allow_none=True)
    protocol_name: str | None = traitlets.Unicode(allow_none=True)

    @staticmethod
    def from_file(path: pathlib.Path) -> '_SeriesMetadata | Error':
        try:
            file_dataset = pydicom.dcmread(path)
        except InvalidDicomError as e:
            return Error(f"Path '{str(path)}' not a valid DICOM file: {e}")

        if "SeriesInstanceUID" in file_dataset:
            uid = str(file_dataset["SeriesInstanceUID"].value)
        else:
            uid = None
        if "SeriesNumber" in file_dataset:
            number = int(file_dataset["SeriesNumber"].value)
        else:
            number = None
        if "SeriesDescription" in file_dataset:
            description = str(file_dataset["SeriesDescription"].value)
        else:
            description = None
        if "ProtocolName" in file_dataset:
            protocol_name = str(file_dataset["ProtocolName"].value)
        else:
            protocol_name = None
        try:
            return _SeriesMetadata(uid=uid, number=number, description=description, protocol_name=protocol_name)
        except traitlets.TraitError as err:
            return Error(f"Failed to read CT slice at path '{str(path)}': {err}")


class _Slice(traitlets.HasTraits):
    file_dataset: pydicom.FileDataset = traitlets.Instance(pydicom.FileDataset, allow_none=False)
    rescale_slope: float = traitlets.Float(allow_none=False)
    rescale_intercept: float = traitlets.Float(allow_none=False)
    rescale_type: str | None = traitlets.Unicode(allow_none=True)
    pixel_spacing: list[float] = traitlets.List(trait=traitlets.Float(allow_none=False), minlen=2, maxlen=2,
                                                allow_none=False)
    image_position_patient: list[float] | None = traitlets.List(trait=traitlets.Float(allow_none=False), minlen=3,
                                                                maxlen=3, allow_none=True)
    series_metadata: _SeriesMetadata = traitlets.Instance(_SeriesMetadata, allow_none=False)

    def convert_to_output_tensor(self, dtype: torch.dtype, **tensor_kwargs) -> torch.Tensor:
        return torch.tensor(pydicom.pixels.pixel_array(self.file_dataset), dtype=dtype,
                            **tensor_kwargs) * self.rescale_slope + self.rescale_intercept

    @staticmethod
    def from_file(path: pathlib.Path) -> '_Slice | Error':
        try:
            file_dataset = pydicom.dcmread(path)
        except InvalidDicomError as e:
            return Error(f"Path '{str(path)}' not a valid DICOM file: {e}")

        if "PixelData" not in file_dataset:
            return Error(f"Field 'PixelData' not present in DICOM file '{str(path)}'.")
        if "RescaleSlope" in file_dataset:
            rescale_slope = float(file_dataset["RescaleSlope"].value)
        else:
            return Error(f"Field 'RescaleSlope' not present in DICOM file '{str(path)}'.")
        if "RescaleIntercept" in file_dataset:
            rescale_intercept = float(file_dataset["RescaleIntercept"].value)
        else:
            return Error(f"Field 'RescaleIntercept' not present in DICOM file '{str(path)}'.")
        if "RescaleType" in file_dataset:
            rescale_type = str(file_dataset["RescaleType"].value)
        else:
            rescale_type = None
        if "PixelSpacing" in file_dataset:
            pixel_spacing = [float(e) for e in file_dataset["PixelSpacing"].value]
        else:
            return Error(f"Field 'PixelSpacing' not present in DICOM file '{str(path)}'.")
        if "ImagePositionPatient" in file_dataset:
            image_position_patient = [float(e) for e in file_dataset["ImagePositionPatient"].value]
        else:
            image_position_patient = None
        series_metadata = _SeriesMetadata.from_file(path)
        if isinstance(series_metadata, Error):
            return Error(f"Failed to read CT slice due to series metadata: {series_metadata.description}")
        try:
            return _Slice(file_dataset=file_dataset, rescale_slope=rescale_slope, rescale_intercept=rescale_intercept,
                          rescale_type=rescale_type, pixel_spacing=pixel_spacing,
                          image_position_patient=image_position_patient, series_metadata=series_metadata)
        except traitlets.TraitError as err:
            return Error(f"Failed to read CT slice at path '{str(path)}': {err}")


class DICOMVolumeLoader(VolumeLoader):
    @staticmethod
    def name() -> str:
        return "DICOM"

    @staticmethod
    def series_available(path: pathlib.Path) -> dict[str, SeriesDescription] | OneSeries:
        if not path.is_dir():
            return {}
        slice_paths = DICOMVolumeLoader._get_slice_paths(path)
        slice_series: list[_SeriesMetadata] = [  #
            series_metadata  #
            for s in slice_paths  #
            if not isinstance(series_metadata := _SeriesMetadata.from_file(s), Error)  #
        ]
        if not slice_series:
            return {}
        # either all slices must have a series UID, or none of them must have one
        if slice_series[0].uid is None:
            if any(s.uid for s in slice_series):
                logger.warning(f"Found some DICOM files with series UIDs and some without in directory '{str(path)}'.")
                return {}
            return OneSeries(file_type=DICOMVolumeLoader.name())
        if not all(s.uid for s in slice_series):
            logger.warning(f"Found some DICOM files with series UIDs and some without in directory '{str(path)}'.")
            return {}
        ret: dict[str, SeriesDescription] = {}
        for s in slice_series:
            if s.uid in ret:
                ret[s.uid].slice_count += 1
            else:
                try:
                    ret[s.uid] = SeriesDescription(file_type=DICOMVolumeLoader.name(), uid=s.uid, slice_count=1,
                                                   number=s.number, description=s.description,
                                                   protocol_name=s.protocol_name)
                except TraitError as e:
                    logger.warning(f"Invalid values found in DICOM series metadata for DICOM '{str(s)}': {e}")
        return ret

    @staticmethod
    def load(path: pathlib.Path, series: str | None) -> Volume | Error:
        slice_paths = DICOMVolumeLoader._get_slice_paths(path)
        slices: list[_Slice] = [  #
            s for slice_path in tqdm(slice_paths, desc=f"Reading DICOM files")  #
            if not isinstance(s := _Slice.from_file(slice_path), Error) and s.series_metadata.uid == series  #
        ]
        if not slices:
            return Error(f"Failed to open any slices from series no. {series} in directory '{str(path)}'.")
        # Filter for slices for which we have "ImagePositionPatient". We will revert back to the original slices
        # if the positioned slices are not evenly spaced.
        positioned_slices = [s for s in slices if s.image_position_patient is not None]

        # Extract the positions of the slices in the x-y plane
        xy_positions = torch.tensor([s.image_position_patient[0:2] for s in positioned_slices])
        # Group these
        slice_index_groupings: list[torch.Tensor] = _extract_groupings(xy_positions, max_cluster_radius=0.5)
        if len(slice_index_groupings) > 1:
            logger.info(f"Found {len(slice_index_groupings)} group(s) of slices in the x-y plane.")
            group_chosen = max(slice_index_groupings, key=lambda t: t.numel())
            logger.info(f"Using largest group which contains {group_chosen.numel()} slices ("
                        f"{100.0 * float(group_chosen.numel()) / float(len(slices))}% of the valid {len(slices)} in "
                        f"the "
                        f"directory).")
        else:
            group_chosen = slice_index_groupings[0]
        # Extract the slices based on the indices
        positioned_slices = [positioned_slices[i] for i in group_chosen.tolist()]
        # Sort slices by z-position
        positioned_slices.sort(key=lambda s: s.image_position_patient[2])

        # Re-extract the slice positions, and then calculate the spacings between the slices
        slice_positions = torch.tensor([s.image_position_patient for s in positioned_slices])
        slice_spacings = torch.linalg.vector_norm(slice_positions[1:, :] - slice_positions[:-1, :], dim=-1)
        # Check for spacings of zero, and remove those slices
        zero_spacings = torch.isclose(slice_spacings, torch.zeros(1), atol=1.0e-3)
        zero_spacing_indices = zero_spacings.nonzero().squeeze(1)
        if len(zero_spacing_indices) > 0:
            logger.warning(f"Removing {len(zero_spacing_indices)} slices, as they have zero spacing.")
            positioned_slices = [positioned_slices[i] for i in range(len(positioned_slices)) if
                                 i not in zero_spacing_indices]
            slice_positions = torch.tensor([s.image_position_patient for s in positioned_slices])
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
        pixel_spacing = slices[0].pixel_spacing
        spacing = torch.tensor([pixel_spacing[1],  # column spacing (x-direction)
                                pixel_spacing[0],  # row spacing (y-direction)
                                slice_spacing  # slice spacing (z-direction)
                                ])
        volume = torch.stack([s.convert_to_output_tensor(torch.float32) for s in slices])
        # logger.info("{}".format(slices[0]["ImageOrientationPatient"]))
        # return volume, spacing, torch.tensor(slices[0].image_position_patient), slices[0].uid
        series_uid = str(path) if series is None else series

        if slices[0].image_position_patient is None:
            logger.info(f"No ImagePositionPatient found for series '{series_uid}' at path '{str(path)}'.")
            ipp = None
        else:
            if (l := len(slices[0].image_position_patient)) == 3:
                ipp = torch.tensor(slices[0].image_position_patient, dtype=torch.float64)
            else:
                logger.info(f"ImagePositionPatient for series '{series_uid}' at path '{str(path)}' is invalid length; "
                            f"expected 3, got {l}.")
                ipp = None

        return Volume(  #
            uid=series_uid,  #
            raw_data=volume,  #
            rescale_slope=slices[0].rescale_slope,  #
            rescale_intercept=slices[0].rescale_intercept,  #
            rescale_type=slices[0].rescale_type,  #
            spacing=spacing,  #
            image_position_patient=ipp  #
        )

    @staticmethod
    def _get_slice_paths(path: pathlib.Path) -> list[pathlib.Path]:
        return [f for f in path.iterdir() if f.is_file() and (f.suffix == ".dcm" or f.suffix == "")]
