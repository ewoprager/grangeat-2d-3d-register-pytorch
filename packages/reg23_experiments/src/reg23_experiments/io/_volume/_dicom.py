import pathlib

import pydicom
import torch
import traitlets
from tqdm import tqdm

from reg23_experiments.data.structs import Error
from ._loader_base import Volume, VolumeLoader

__all__ = ["DICOMVolumeLoader"]


class CTSliceDICOM(traitlets.HasTraits):
    file_dataset: pydicom.FileDataset = traitlets.Instance(pydicom.FileDataset, allow_none=False)
    series_number: int | None = traitlets.Int(allow_none=True)
    rescale_slope: float = traitlets.Float(allow_none=False)
    rescale_intercept: float = traitlets.Float(allow_none=False)
    rescale_type: str | None = traitlets.Unicode(allow_none=True)
    pixel_spacing: list[float] = traitlets.List(trait=traitlets.Float(allow_none=False), minlen=2, maxlen=2,
                                                allow_none=False)
    image_position_patient: list[float] = traitlets.List(trait=traitlets.Float(allow_none=False), minlen=3, maxlen=3,
                                                         allow_none=True)
    uid: str = traitlets.Unicode(allow_none=False)

    def convert_to_output_tensor(self, dtype: torch.dtype, **tensor_kwargs) -> torch.Tensor:
        return torch.tensor(pydicom.pixels.pixel_array(self.file_dataset), dtype=dtype,
                            **tensor_kwargs) * self.rescale_slope + self.rescale_intercept


def read_ct_slice_dicom(path: pathlib.Path) -> CTSliceDICOM | Error:
    file_dataset = pydicom.dcmread(path)
    if "PixelData" not in file_dataset:
        return Error(f"Field 'PixelData' not present in DICOM file '{str(path)}'.")
    if "StudyInstanceUID" in file_dataset:
        if "SeriesInstanceUID" in file_dataset:
            uid = file_dataset["StudyInstanceUID"].value + "/" + file_dataset["SeriesInstanceUID"].value
        else:
            return Error(f"Field 'SeriesInstanceUID' not present in DICOM file '{str(path)}'.")
    else:
        return Error(f"Field 'StudyInstanceUID' not present in DICOM file '{str(path)}'.")
    if "SeriesNumber" in file_dataset:
        series_number = int(file_dataset["SeriesNumber"].value)
    else:
        series_number = None
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
    try:
        return CTSliceDICOM(file_dataset=file_dataset, series_number=series_number, rescale_slope=rescale_slope,
                            rescale_intercept=rescale_intercept, rescale_type=rescale_type, pixel_spacing=pixel_spacing,
                            image_position_patient=image_position_patient, uid=uid)
    except traitlets.TraitError as err:
        return Error(f"Failed to read CT slice at path '{str(path)}': {err}")


class DICOMVolumeLoader(VolumeLoader):
    def series_available(self, path: pathlib.Path) -> list[int]:
        if not path.is_dir():
            return []
        all_slice_paths = [f for f in path.iterdir() if f.is_file() and (f.suffix == ".dcm" or f.suffix == "")]
        if len(all_slice_paths) == 0:
            return []
        all_slices: list[CTSliceDICOM] = [  #
            s for slice_path in tqdm(all_slice_paths, desc="Reading DICOM files")  #
            if not isinstance(s := read_ct_slice_dicom(slice_path), Error)  #
        ]
        return list(set([(-1 if s.series_number is None else s.series_number) for s in all_slices]))

    def load(self, path: pathlib.Path, series: int) -> Volume | Error:
        all_slice_paths = [elem for elem in path.iterdir() if elem.is_file()]
    if check_for_dcm_suffix:
        all_slice_paths = [elem for elem in all_slice_paths if elem.suffix == ".dcm"]
    if len(all_slice_paths) == 0:
        raise Exception(
            "Error: no {}slices found in given directory '{}'.".format("DICOM (.dcm) " if check_for_dcm_suffix else "",
                                                                       str(path)))
    slices: list[CTSliceDICOM] = [  #
        s for slice_path in tqdm(all_slice_paths, desc=f"Reading DICOM files")  #
        if (s := read_ct_slice_dicom(slice_path)) is not None  #
    ]
    if series_number is not None:
        slices = [s for s in slices if s.series_number == series_number]

    if not slices:
        raise Exception("Error: failed to open any {}slices found in given directory '{}'{}.".format(
            "DICOM (.dcm) " if check_for_dcm_suffix else "", str(path), series_number,
            "" if series_number is None else f" in series {series_number}"))
    # Filter for slices for which we have "ImagePositionPatient". We will revert back to the original slices
    # if the positioned slices are not evenly spaced.
    positioned_slices = [s for s in slices if s.image_position_patient is not None]

    # Extract the positions of the slices in the x-y plane
    xy_positions = torch.tensor([s.image_position_patient[0:2] for s in positioned_slices])
    # Group these
    slice_index_groupings: list[torch.Tensor] = extract_groupings(xy_positions, max_cluster_radius=0.5)
    if len(slice_index_groupings) > 1:
        logger.info(f"Found {len(slice_index_groupings)} group(s) of slices in the x-y plane.")
        group_chosen = max(slice_index_groupings, key=lambda t: t.numel())
        logger.info(f"Using largest group which contains {group_chosen.numel()} slices ("
                    f"{100.0 * float(group_chosen.numel()) / float(len(slices))}% of the valid {len(slices)} in the "
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
        logger.warning("Median slice spacing is {:.4f}; some slices deviate from this:".format(slice_spacing.item()))
        for index in bad_spacing_indices:
            logger.warning(
                "Spacing between slices {} and {} is {:.4f}".format(index, index + 1, slice_spacings[index].item()))
        logger.warning("Slice positioning was inconsistent, so reverting to using all slices (regardless of whether "
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
    return volume, spacing, torch.tensor(slices[0].image_position_patient), slices[0].uid
        return Volume(raw_data=volume, spacing=spacing)
