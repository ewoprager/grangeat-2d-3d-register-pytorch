from typing import NamedTuple, Type

from registration.lib import sinogram
from registration.lib.structs import Transformation


class DrrVsGrangeatPlotData(NamedTuple):
    class Dataset(NamedTuple):
        ct_volume_numel: int
        sinogram3d_size: int
        sinogram_type: Type[sinogram.SinogramType]
        x_ray_numel: int
        sinogram2d_size: int
        drr_time: float
        resample_time: float
        grangeat_fixed_image_time: float

    datasets: list[Dataset]


class RegisterPlotData(NamedTuple):
    class Dataset(NamedTuple):
        fixed_image_numel: int
        obj_func_name: str
        sinogram_type: Type[sinogram.SinogramType]
        time_per_iteration: float
        ground_truth_transformation: Transformation
        starting_transformation: Transformation
        converged_transformation: Transformation

    iteration_count: int
    particle_count: int
    datasets: list[Dataset]
