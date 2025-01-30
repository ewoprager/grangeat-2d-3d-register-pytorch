from typing import NamedTuple, Tuple

import torch

class LinearMapping:
    def __init__(self, a: float | torch.Tensor, b: float | torch.Tensor):
        self.a = a
        self.b = b

    def __call__(self, x: float | torch.Tensor) -> float | torch.Tensor:
        return self.a + self.b * x


class LinearRange:
    def __init__(self, low: float, high: float, ):
        self.low = low
        self.high = high

    def generate_range(self, count: int, *, device) -> torch.Tensor:
        return torch.linspace(self.low, self.high, count, device=device)

    def get_mapping_from(self, other: 'LinearRange') -> LinearMapping:
        frac: float = (self.high - self.low) / (other.high - other.low)
        return LinearMapping(self.low - frac * other.low, frac)


class Transformation(NamedTuple):
    rotation: torch.Tensor
    translation: torch.Tensor

    def inverse(self) -> 'Transformation':
        return Transformation(-self.rotation, -self.translation)


class SceneGeometry(NamedTuple):
    source_distance: float  # [mm]; distance in the positive z-direction from the centre of the detector array
    ct_origin_distance: float  # [mm]; distance in the positive z-direction from the centre of the detector array)


class Sinogram2dGrid(NamedTuple):
    phi: torch.Tensor
    r: torch.Tensor

    def device_consistent(self) -> bool:
        return self.phi.device == self.r.device


class Sinogram3dGrid(NamedTuple):
    phi: torch.Tensor
    theta: torch.Tensor
    r: torch.Tensor

    def device_consistent(self) -> bool:
        return self.phi.device == self.theta.device and self.theta.device == self.r.device


class Sinogram2dRange(NamedTuple):
    phi: LinearRange
    r: LinearRange

    def generate_linear_grid(self, counts: int | Tuple[int] | torch.Size, *, device=torch.device("cpu")):
        if isinstance(counts, int):
            counts = (counts, counts)
        return Sinogram2dGrid(torch.linspace(self.phi.low, self.phi.high, counts[0], device=device),
                              torch.linspace(self.r.low, self.r.high, counts[1], device=device))


class Sinogram3dRange(NamedTuple):
    phi: LinearRange
    theta: LinearRange
    r: LinearRange

    def generate_linear_grid(self, counts: int | Tuple[int] | torch.Size, *, device=torch.device("cpu")):
        if isinstance(counts, int):
            counts = (counts, counts, counts)
        return Sinogram3dGrid(torch.linspace(self.phi.low, self.phi.high, counts[0], device=device),
                              torch.linspace(self.theta.low, self.theta.high, counts[1], device=device),
                              torch.linspace(self.r.low, self.r.high, counts[2], device=device))


class VolumeSpec(NamedTuple):
    ct_volume_path: str
    downsample_factor: int
    sinogram: torch.Tensor
    sinogram_range: Sinogram3dRange


class DrrSpec(NamedTuple):
    ct_volume_path: str
    detector_spacing: torch.Tensor  # [mm] distances between the detectors: (vertical, horizontal)
    scene_geometry: SceneGeometry
    image: torch.Tensor
    sinogram: torch.Tensor
    sinogram_range: Sinogram2dRange
    transformation: Transformation