import logging
import pathlib
from typing import Tuple

import nibabel
import nrrd
import pydicom
import torch
from tqdm import tqdm
import traitlets

from reg23_experiments.data import sinogram
from reg23_experiments.data.structs import LinearRange, SceneGeometry, Transformation
from reg23_experiments.ops.volume import fit_line_3d, point_line_distance_3d

__all__ = ["read_nrrd", "read_nii", "read_dicom_directory_as_volume", "read_volume", "load_ct", "load_cached_ct"]

logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals(
    [sinogram.VolumeSpec, sinogram.SinogramClassic, sinogram.SinogramHEALPix, LinearRange, SceneGeometry,
     Transformation])


def read_nrrd(path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor]:
    data, header = nrrd.read(str(path))
    data = torch.tensor(data)
    directions = torch.tensor(header['space directions'], dtype=torch.float32)
    spacing = directions.norm(dim=1).flip(dims=(0,))
    return data, spacing


def read_nii(path: pathlib.Path) -> tuple[torch.Tensor, torch.Tensor]:
    img = nibabel.load(str(path))
    data = torch.tensor(img.get_fdata())
    data = data.permute(*reversed(range(data.ndim)))
    spacing = torch.tensor(img.header.get_zooms(), dtype=torch.float32)[0:3]
    return data, spacing


class CTSliceDICOM(traitlets.HasTraits):
    file_dataset: pydicom.FileDataset = traitlets.Instance(pydicom.FileDataset, allow_none=False)
    rescale_slope: float = traitlets.Float(allow_none=False)
    rescale_intercept: float = traitlets.Float(allow_none=False)
    rescale_type: str = traitlets.Unicode(allow_none=False)
    pixel_spacing: list[float] = traitlets.List(trait=traitlets.Float(allow_none=False), minlen=2, maxlen=2,
                                                allow_none=False)
    image_position_patient: list[float] = traitlets.List(trait=traitlets.Float(allow_none=False), minlen=3, maxlen=3,
                                                         allow_none=True)

    def convert_to_output_tensor(self, dtype: torch.dtype, **tensor_kwargs) -> torch.Tensor:
        return torch.tensor(pydicom.pixels.pixel_array(self.file_dataset), dtype=dtype,
                            **tensor_kwargs) * self.rescale_slope + self.rescale_intercept


def read_ct_slice_dicom(path: pathlib.Path) -> CTSliceDICOM | None:
    file_dataset = pydicom.dcmread(path)
    if "PixelData" not in file_dataset:
        return None
    if "RescaleSlope" in file_dataset:
        rescale_slope = float(file_dataset["RescaleSlope"].value)
    else:
        return None
    if "RescaleIntercept" in file_dataset:
        rescale_intercept = float(file_dataset["RescaleIntercept"].value)
    else:
        return None
    if "RescaleType" in file_dataset:
        rescale_type = str(file_dataset["RescaleType"].value)
    else:
        return None
    if "PixelSpacing" in file_dataset:
        pixel_spacing = [float(e) for e in file_dataset["PixelSpacing"].value]
    else:
        return None
    if "ImagePositionPatient" in file_dataset:
        image_position_patient = [float(e) for e in file_dataset["ImagePositionPatient"].value]
    else:
        image_position_patient = None
    try:
        return CTSliceDICOM(file_dataset=file_dataset, rescale_slope=rescale_slope, rescale_intercept=rescale_intercept,
                            rescale_type=rescale_type, pixel_spacing=pixel_spacing,
                            image_position_patient=image_position_patient)
    except traitlets.TraitError as err:
        logger.error(f"Failed to read CT slice at path '{str(path)}': {err}")
        return None


def extract_groupings(points: torch.Tensor, *, max_cluster_radius: float) -> list[torch.Tensor]:
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


def read_dicom_directory_as_volume(path: pathlib.Path, *, check_for_dcm_suffix: bool = True) -> tuple[
    torch.Tensor, torch.Tensor]:
    slice_paths = [elem for elem in path.iterdir() if elem.is_file()]
    if check_for_dcm_suffix:
        slice_paths = [elem for elem in slice_paths if elem.suffix == ".dcm"]
    if len(slice_paths) == 0:
        raise Exception("Error: no {}slice_paths found in given directory '{}'.".format(
            "DICOM (.dcm) " if check_for_dcm_suffix else "", str(path)))
    slices: list[CTSliceDICOM] = [  #
        s for slice_path in tqdm(slice_paths, desc=f"Reading DICOM files")  #
        if (s := read_ct_slice_dicom(slice_path)) is not None  #
    ]
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

    if False:
        # Extract the positions from the slices
        slice_positions = torch.tensor([s.image_position_patient for s in positioned_slices])

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
        def distance_of_slice(s: CTSliceDICOM) -> float:
            position = torch.tensor(s.image_position_patient)
            return torch.dot(position, line_direction).item()

        positioned_slices.sort(key=distance_of_slice)
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
    return volume, spacing


def read_volume(path: pathlib.Path, *, check_for_dcm_suffix_if_dir: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    if path.is_file():
        logger.info("Loading CT data file '{}'...".format(str(path)))
        # Obtain a data tensor and a tensor of voxel spacing
        if path.suffix == ".nrrd":
            data, spacing = read_nrrd(path)
            logger.warning("Don't know how to read ImageOrientationPatient from .nrrd files.")
        elif path.suffix == ".nii":
            data, spacing = read_nii(path)
            logger.warning("Don't know how to read ImageOrientationPatient from .nii files.")
        else:
            raise Exception(f"Error: file {str(path)} is of unrecognised type.")
        # Make sure there aren't unnecessary dimensions
        if len(data.size()) > 3:
            data = data.squeeze()
        if len(data.size()) != 3:
            raise Exception(f"Error: CT volume file '{str(path)}' contains invalid size '{str(data.size())}'")
        return data, spacing
    if path.is_dir():
        logger.info(f"Loading CT DICOM data from directory '{str(path)}'")
        return read_dicom_directory_as_volume(path, check_for_dcm_suffix=check_for_dcm_suffix_if_dir)
    raise Exception(f"Given path '{str(path)}' is not a file or directory.")


def load_ct(path: pathlib.Path, *, hu_cutoff: float = -1000.0, mu_water: float = 0.02,
            check_for_dcm_suffix_if_dir: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a CT volume from a file or directory

    :param path: Path to a .nrrd or .nii file (a CT volume) or a directory containing multiple .dcm files (one for each
    slice of a CT volume)
    :param hu_cutoff: The cutoff lower bound in Hounsfield units to clamp to.
    :param mu_water: The attenuation coefficient of water in [mm^-1]
    :return: (The CT volume, a tensor of size (3,): the spacing of the volume voxels in order XYZ)
    :rtype: tuple[torch.Tensor, torch.Tensor], (the CT volume of attenuation coefficient, voxel spacing in [mm])
    """
    data, spacing = read_volume(path, check_for_dcm_suffix_if_dir=check_for_dcm_suffix_if_dir)
    sizes = data.size()
    spacing = spacing
    logger.info("CT data volume size and spacing = [{} x {} x {}]; [{:.4f} x {:.4f} x {:.4f}] mm"
                "".format(sizes[0], sizes[1], sizes[2], spacing[0], spacing[1], spacing[2]))
    if path.name == "PhantomCT.nii":
        # ToDo: Double check this:
        logger.warning("Applying super hacky adjustment; check it's correct!")
        data -= 1000.0  # super hacky adjustment for one particular file which appears to be HU + 1000
    image_hu = data.to(dtype=torch.float32)
    image_hu[image_hu < hu_cutoff] = hu_cutoff
    image_mu = mu_water * (1.0 + image_hu / 1000.0)

    return image_mu, spacing


def load_cached_ct(cache_directory: str, sinogram_hash: str) -> Tuple[int, sinogram.Sinogram] | None:
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
