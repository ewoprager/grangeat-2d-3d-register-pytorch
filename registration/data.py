import torch
import nrrd

from registration.lib.structs import *


def read_nrrd(path: str, downsample_factor=1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print("Loading CT data file {}...".format(path))
    data, header = nrrd.read(path)
    print("Done.")
    print("Processing CT data...")
    sizes = header['sizes']
    print("\tVolume size = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    data = torch.tensor(data, device="cpu")
    image = torch.maximum(data.type(torch.float32) + 1000., torch.tensor([0.], device=data.device))
    if downsample_factor > 1:
        down_sampler = torch.nn.AvgPool3d(downsample_factor)
        image = down_sampler(image[None, :, :, :])[0]
    sizes = image.size()
    print("\tVolume size after down-sampling = [{} x {} x {}]".format(sizes[0], sizes[1], sizes[2]))
    bounds = torch.Tensor([image.min().item(), image.max().item()])
    print("\tValue range = ({:.3f}, {:.3f})".format(bounds[0], bounds[1]))
    bounds[1] *= 10000.
    directions = torch.tensor(header['space directions'])
    spacing = float(downsample_factor) * directions.norm(dim=1)
    print("\tCT voxel spacing = [{} x {} x {}] mm".format(spacing[0], spacing[1], spacing[2]))
    print("Done.")

    return image, spacing, bounds


def load_cached_volume(cache_directory: str):
    file: str = cache_directory + "/volume_spec.pt"
    try:
        volume_spec = torch.load(file)
    except:
        print("No cache file '{}' found.".format(file))
        return None
    if not isinstance(volume_spec, VolumeSpec):
        print("Cache file '{}' invalid.".format(file))
        return None
    path = volume_spec.ct_volume_path
    volume_downsample_factor = volume_spec.downsample_factor
    sinogram3d = volume_spec.sinogram
    print("Loaded cached volume spec from '{}'".format(file))
    return path, volume_downsample_factor, sinogram3d


def load_cached_volume_fibonacci(cache_directory: str):
    file: str = cache_directory + "/volume_spec_fibonacci.pt"
    try:
        volume_spec = torch.load(file)
    except:
        print("No cache file '{}' found.".format(file))
        return None
    if not isinstance(volume_spec, VolumeSpecFibonacci):
        print("Cache file '{}' invalid.".format(file))
        return None
    path = volume_spec.ct_volume_path
    volume_downsample_factor = volume_spec.downsample_factor
    sinogram3d = volume_spec.sinogram
    r_range = volume_spec.r_range
    print("Loaded cached Fibonacci volume spec from '{}'".format(file))
    return path, volume_downsample_factor, sinogram3d, r_range


def load_cached_drr(cache_directory: str, ct_volume_path: str):
    file: str = cache_directory + "/drr_spec.pt"
    try:
        drr_spec = torch.load(file)
    except:
        print("No cache file '{}' found.".format(file))
        return None
    if not isinstance(drr_spec, DrrSpec):
        print("Cache file '{}' invalid.".format(file))
        return None
    if drr_spec.ct_volume_path != ct_volume_path:
        print("Cached drr '{}' is from different volume = '{}'; required volume = {}.".format(file,
                                                                                              drr_spec.ct_volume_path,
                                                                                              ct_volume_path))
        return None
    detector_spacing = drr_spec.detector_spacing
    scene_geometry = drr_spec.scene_geometry
    drr_image = drr_spec.image
    fixed_image = drr_spec.sinogram
    sinogram2d_range = drr_spec.sinogram_range
    transformation_ground_truth = drr_spec.transformation
    print("Loaded cached drr spec from '{}'".format(file))
    return detector_spacing, scene_geometry, drr_image, fixed_image, sinogram2d_range, transformation_ground_truth
