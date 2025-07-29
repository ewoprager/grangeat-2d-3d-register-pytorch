from typing import NamedTuple

import numpy as np

class DrrVsGrangeatPlotData(NamedTuple):
    class Dataset(NamedTuple):
        ct_volume_numel: int
        sinogram3d_size: int
        x_ray_numel: int
        sinogram2d_size: int
        drr_time: float
        resample_time: float

    datasets: list[Dataset]
