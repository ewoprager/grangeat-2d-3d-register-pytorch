from typing import NamedTuple, Type

from registration.lib import sinogram

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
