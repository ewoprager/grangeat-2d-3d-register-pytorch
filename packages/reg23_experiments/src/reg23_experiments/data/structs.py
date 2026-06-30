from abc import ABC, abstractmethod
from typing import NamedTuple, Sequence, Tuple, Union

import kornia
import numpy
import scipy
import torch
import traitlets
from beartype import beartype as typechecker
from jaxtyping import Float32, Float64, jaxtyped

__all__ = ["Error", "GrowingTensor", "LinearMapping", "LinearRange", "Transformation", "SceneGeometry", "Cropping",
           "Sinogram2dRange", "Sinogram2dGrid", "Sinogram3dGrid", "OptimisationInstance"]


class Error:
    def __init__(self, description: str):
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    def __str__(self) -> str:
        return f"{self._description}"

    def __repr__(self) -> str:
        return f"Error(description='{self._description}')"


class GrowingTensor:
    def __init__(self, element_shape: Sequence[int], initial_length: int, **kwargs):
        self.element_size = torch.Size(element_shape)
        self.data = torch.zeros([initial_length] + list(element_shape), **kwargs)
        self.count = 0

    def push_back(self, element: torch.Tensor) -> None:
        assert element.size() == self.element_size
        if self.count >= self.data.size()[0]:
            self.data = torch.cat((self.data, torch.zeros_like(self.data)), dim=0)
        self.data[self.count] = element.to(dtype=self.data.dtype, device=self.data.device)
        self.count += 1

    def get(self) -> torch.Tensor:
        return self.data[:self.count]


class LinearMapping:
    def __init__(self, intercept: float | torch.Tensor, gradient: float | torch.Tensor):
        self.intercept = intercept
        self.gradient = gradient

    def __call__(self, x: float | torch.Tensor) -> float | torch.Tensor:
        return self.intercept + self.gradient * x


class LinearRange:
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def generate_grid(self, count: int, *, device=torch.device('cpu')) -> torch.Tensor:
        return torch.linspace(self.low, self.high, count, device=device)

    def generate_tex_coord_grid(self, count: int, *, device=torch.device('cpu')) -> torch.Tensor:
        half_cell_size = 0.5 * (self.high - self.low) / float(count)
        return torch.linspace(self.low + half_cell_size, self.high - half_cell_size, count, device=device)

    def get_mapping_from(self, other: 'LinearRange') -> LinearMapping:
        frac: float = (self.high - self.low) / (other.high - other.low)
        return LinearMapping(self.low - frac * other.low, frac)

    def get_spacing(self, count: int) -> float:
        return (self.high - self.low) / float(count - 1)

    def get_tex_coord_spacing(self, count: int) -> float:
        return (self.high - self.low) / float(count)

    def get_centre(self) -> float:
        return .5 * (self.low + self.high)

    @classmethod
    def grid_sample_range(cls):
        return LinearRange(-1., 1.)


class Transformation:
    @jaxtyped(typechecker=typechecker)
    def __init__(self, *, rotation: Float64[torch.Tensor, "3"], translation: Float64[torch.Tensor, "3"]):
        assert rotation.device == translation.device
        self._rotation = rotation
        self._translation = translation

    @property
    def rotation(self) -> Float64[torch.Tensor, "3"]:
        return self._rotation

    @property
    def translation(self) -> Float64[torch.Tensor, "3"]:
        return self._translation

    def inverse(self) -> 'Transformation':
        r_inverse = kornia.geometry.conversions.axis_angle_to_rotation_matrix(-self.rotation.unsqueeze(0))[0]
        r_inverse_t = torch.einsum('kl,...l->...k', r_inverse, self.translation.unsqueeze(0))[0]
        return Transformation(rotation=-self.rotation, translation=-r_inverse_t)

    @jaxtyped(typechecker=typechecker)
    def get_h(self, device: torch.device) -> Float64[torch.Tensor, "4 4"]:
        """
        :param device: The device to put the returned tensor on
        :return: [(4, 4) tensor] The homogenous affine transformation matrix H corresponding to this transformation.
        Stored column-major.
        """
        r = kornia.geometry.conversions.axis_angle_to_rotation_matrix(self.rotation.unsqueeze(0))[0].to(device=device)
        rt = torch.hstack([r, self.translation.to(device=device).t().unsqueeze(-1)])
        return torch.vstack([rt, torch.tensor([0., 0., 0., 1.], device=device, dtype=torch.float64).unsqueeze(0)])

    def vectorised(self) -> Float64[torch.Tensor, "6"]:
        return torch.cat((self.rotation, self.translation), dim=0)

    @jaxtyped(typechecker=typechecker)
    def with_translation_offset(self, translation_offset: Float64[torch.Tensor, "2"]) -> 'Transformation':
        return Transformation(  #
            rotation=self.rotation.clone(),  #
            translation=self.translation + torch.cat((  #
                translation_offset.to(device=self.device),  #
                torch.zeros(1, dtype=torch.float64, device=self.device)  #
            ))  #
        )

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def from_vector(vector: Float64[torch.Tensor, "6"]) -> 'Transformation':
        return Transformation(rotation=vector[0:3], translation=vector[3:6])

    @property
    def device(self):
        return self.rotation.device

    def to(self, **kwargs) -> 'Transformation':
        return Transformation(rotation=self.rotation.to(**kwargs), translation=self.translation.to(**kwargs))

    def clone(self) -> 'Transformation':
        return Transformation(rotation=self.rotation.clone(), translation=self.translation.clone())

    def distance(self, other: 'Transformation') -> float:
        device = self.translation.device
        r1 = kornia.geometry.conversions.axis_angle_to_rotation_matrix(  #
            self.rotation.unsqueeze(0))[0].to(device=device, dtype=torch.float64)
        r2 = kornia.geometry.conversions.axis_angle_to_rotation_matrix(  #
            other.rotation.unsqueeze(0))[0].to(device=device, dtype=torch.float64)
        return (((self.translation - other.translation) / 100.).square().sum() + torch.tensor(
            numpy.array([numpy.real(scipy.linalg.logm((torch.matmul(r1.t(), r2).cpu().numpy())))]), dtype=torch.float64,
            device=device).square().sum()).sqrt().item()

    def is_close(self, other: 'Transformation') -> bool:
        return torch.allclose(self.rotation, other.rotation.to(device=self.rotation.device)) and torch.allclose(
            self.translation, other.translation.to(device=self.translation.device))

    def __str__(self) -> str:
        return "Transformation(rot = {}, trans = {})".format(str(self.rotation), str(self.translation))

    @classmethod
    def zero(cls, device: torch.device = torch.device("cpu")) -> 'Transformation':
        return Transformation(rotation=torch.zeros(3, dtype=torch.float64, device=device),
                              translation=torch.zeros(3, dtype=torch.float64, device=device))

    @classmethod
    def random_uniform(cls, device: torch.device = torch.device("cpu")) -> 'Transformation':
        return Transformation(  #
            rotation=torch.pi * (-1. + 2. * torch.rand(3, dtype=torch.float64, device=device)),  #
            translation=25. * (-1. + 2. * torch.rand(3, dtype=torch.float64, device=device)) + Transformation.zero(
                device).translation  #
        )

    @classmethod
    def random_gaussian(cls, *, rotation_mean: torch.Tensor, rotation_std: Union[torch.Tensor, float],
                        translation_mean: torch.Tensor, translation_std: Union[torch.Tensor, float],
                        generator=None) -> 'Transformation':
        assert rotation_mean.size() == torch.Size([3])
        assert translation_mean.size() == torch.Size([3])
        assert translation_mean.device == rotation_mean.device
        assert isinstance(rotation_std, float) or (
                (rotation_std.size() == torch.Size([3])) and (rotation_std.device == rotation_mean.device))
        assert isinstance(translation_std, float) or (
                (translation_std.size() == torch.Size([3])) and (translation_std.device == rotation_mean.device))
        return Transformation(rotation=torch.normal(mean=rotation_mean, std=rotation_std, generator=generator),
                              translation=torch.normal(mean=translation_mean, std=translation_std, generator=generator))


class SceneGeometry(NamedTuple):
    source_distance: float  # [mm]; distance in the positive z-direction from the centre of the detector array
    fixed_image_offset: Float64[torch.Tensor, "2"] = torch.zeros(2, dtype=torch.float64)  # size (2,): (x,

    # y) [mm]; offset of the fixed image relative to the source

    @jaxtyped(typechecker=typechecker)
    def source_position(self, device: torch.device) -> Float64[torch.Tensor, "3"]:
        return torch.tensor([0., 0., self.source_distance], device=device, dtype=torch.float64)

    @classmethod
    @jaxtyped(typechecker=typechecker)
    def projection_matrix(cls, source_position: Float64[torch.Tensor, "3"],
                          central_ray: Float64[torch.Tensor, "3"] | None = None) -> Float64[torch.Tensor, "4 4"]:
        # ToDo: Make this a normal method (rather than a class method) that also takes fixed_image_offset into account
        """
        Generate the projection matrix for the given source position

        :param source_position: [(3,) tensor] the position of the X-ray source
        :param central_ray: [(3,) tensor or None] the vector from the X-ray source to the closest point on the
        detector array. If none, the detector array is assumed to be the x-y plane.
        :return: [(4, 4) tensor] The projection matrix P that projects points in homogeneous coordinates away from
        the given source position onto the plane of the detector array, as given by the central ray.
        """
        device = source_position.device

        if central_ray is None:
            central_ray: Float64[torch.Tensor, "3"] = torch.tensor([0., 0., - source_position[2]], device=device)

        assert central_ray.device == device
        assert source_position.size() == torch.Size([3])
        assert central_ray.size() == torch.Size([3])

        m_matrix: torch.Tensor = torch.outer(torch.hstack((source_position, torch.tensor([1.], device=device))),
                                             central_ray) + torch.dot(central_ray, central_ray) * torch.vstack(
            (torch.eye(3, device=device), torch.zeros((1, 3), device=device)))

        return torch.hstack((m_matrix, -torch.matmul(m_matrix, source_position.t().unsqueeze(-1))))


class Cropping(traitlets.HasTraits):
    """
    Struct that represents rectangular cropping of an image, by the positions of the right, top, left and bottom edges
    of the rectangle as fractions of the dimensions of the image.

    The positive directions are to the right and down, and fractional distances are given in these positive directions,
    so 'no cropping' is represented by `Cropping(right=1.0, top=0.0, left=0.0, bottom=1.0)` (the default values).

    A valid Cropping must have right > left, and bottom > top.
    """

    right: float = traitlets.Float(default_value=1.0, min=0.0, max=1.0).tag(ui=True)
    top: float = traitlets.Float(default_value=0.0, min=0.0, max=1.0).tag(ui=True)
    left: float = traitlets.Float(default_value=0.0, min=0.0, max=1.0).tag(ui=True)
    bottom: float = traitlets.Float(default_value=1.0, min=0.0, max=1.0).tag(ui=True)

    def get_fractional_centre_offset(self, **tensor_kwargs) -> torch.Tensor:
        return 0.5 * torch.tensor([self.left + self.right, self.top + self.bottom], **tensor_kwargs) - 0.5

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        i0 = int(round(self.left * float(tensor.size(1))))
        i1 = int(round(self.right * float(tensor.size(1))))
        j0 = int(round(self.top * float(tensor.size(0))))
        j1 = int(round(self.bottom * float(tensor.size(0))))
        return tensor[j0:j1, i0:i1]

    def expand_image_fraction(self, image_fraction: float) -> 'Cropping':
        return Cropping(  #
            right=min(1.0, self.right + image_fraction),  #
            top=max(0.0, self.top - image_fraction),  #
            left=max(0.0, self.left - image_fraction),  #
            bottom=min(1.0, self.bottom + image_fraction),  #
        )

    @jaxtyped(typechecker=typechecker)
    def expand_mm(self, distance: float, *, image_size: torch.Size,
                  image_spacing: Float64[torch.Tensor, "2"]) -> 'Cropping':
        h = distance / (float(image_size[1]) * image_spacing[0].item())
        v = distance / (float(image_size[0]) * image_spacing[1].item())
        return Cropping(  #
            right=min(1.0, self.right + h),  #
            top=max(0.0, self.top - v),  #
            left=max(0.0, self.left - h),  #
            bottom=min(1.0, self.bottom + v),  #
        )

    @staticmethod
    def intersect(a: 'Cropping', b: 'Cropping') -> 'Cropping':
        return Cropping(  #
            right=min(a.right, b.right),  #
            top=max(a.top, b.top),  #
            left=max(a.left, b.left),  #
            bottom=min(a.bottom, b.bottom)  #
        )

    @traitlets.observe("left", "right")
    def _check_horizontal(self, change):
        self.left = max(0.0, self.left)
        self.right = min(1.0, self.right)
        if self.left > self.right:
            raise traitlets.TraitError(f"Cropping 'left' value {self.left} exceeds 'right' value {self.right}.")

    @traitlets.observe("top", "bottom")
    def _check_vertical(self, change):
        self.top = max(0.0, self.top)
        self.bottom = min(1.0, self.bottom)
        if self.top > self.bottom:
            raise traitlets.TraitError(f"Cropping 'top' value {self.top} exceeds 'bottom' value {self.bottom}.")


class Sinogram2dRange(NamedTuple):
    phi: LinearRange
    r: LinearRange


class Sinogram2dGrid(NamedTuple):
    phi: Float32[torch.Tensor, "*_"]
    r: Float32[torch.Tensor, "*_"]

    def to(self, *, device: torch.device) -> 'Sinogram2dGrid':
        return Sinogram2dGrid(self.phi.to(device=device), self.r.to(device=device))

    def device_consistent(self) -> bool:
        return self.phi.device == self.r.device

    def size_consistent(self) -> bool:
        return self.phi.size() == self.r.size()

    def shifted(self, offset: torch.Tensor) -> 'Sinogram2dGrid':
        assert offset.size() == torch.Size([2])
        assert self.device_consistent()
        cp = self.phi.cos()
        sp = self.phi.sin()
        unit = torch.stack((cp, sp), dim=-1)
        del cp, sp
        delta = torch.einsum("...i, i -> ...", unit, offset.to(device=unit.device, dtype=unit.dtype))
        del unit
        return Sinogram2dGrid(self.phi, self.r - delta)

    @classmethod
    def linear_from_range(cls, sinogram_range: Sinogram2dRange, counts: int | Tuple[int, int] | torch.Size,
                          **tensor_kwargs) -> 'Sinogram2dGrid':
        if isinstance(counts, int):
            counts = (counts, counts)
        phis = torch.linspace(sinogram_range.phi.low, sinogram_range.phi.high, counts[0], **tensor_kwargs)
        rs = torch.linspace(sinogram_range.r.low, sinogram_range.r.high, counts[1], **tensor_kwargs)
        phis, rs = torch.meshgrid(phis, rs)
        return Sinogram2dGrid(phis, rs)


class Sinogram3dGrid(NamedTuple):
    phi: Float32[torch.Tensor, "*_"]
    theta: Float32[torch.Tensor, "*_"]
    r: Float32[torch.Tensor, "*_"]

    def to(self, *, device: torch.device) -> 'Sinogram3dGrid':
        return Sinogram3dGrid(self.phi.to(device=device), self.theta.to(device=device), self.r.to(device=device))

    def device_consistent(self) -> bool:
        return self.phi.device == self.theta.device and self.theta.device == self.r.device

    def size_consistent(self) -> bool:
        return self.phi.size() == self.theta.size() and self.theta.size() == self.r.size()

    def unflip(self) -> 'Sinogram3dGrid':
        assert self.size_consistent()
        assert self.device_consistent()

        theta_div = torch.div(self.theta + .5 * torch.pi, torch.pi, rounding_mode="floor")
        theta_flip = torch.fmod(theta_div.to(dtype=torch.int32).abs(), 2).to(dtype=torch.bool)
        phi_div = torch.div(self.phi + .5 * torch.pi, torch.pi, rounding_mode="floor")
        phi_flip = torch.fmod(phi_div.to(dtype=torch.int32).abs(), 2).to(dtype=torch.bool)

        ret_theta = self.theta - torch.pi * theta_div

        del theta_div

        ret_phi = self.phi - torch.pi * phi_div

        del phi_div

        ret_theta[torch.logical_and(phi_flip, torch.logical_not(theta_flip))] *= -1.
        ret_r = self.r.clone()
        ret_r[torch.logical_xor(theta_flip, phi_flip)] *= -1.

        del theta_flip, phi_flip

        return Sinogram3dGrid(ret_phi, ret_theta, ret_r)

    # @classmethod  # def fibonacci_from_r_range(cls, r_range: LinearRange, r_count: int, *, spiral_count: int | None
    # = None,  #                            device=torch.device("cpu")) -> 'Sinogram3dGrid':  #     if spiral_count
    # is None:  #         spiral_count = r_count * r_count  #     rs = torch.linspace(r_range.low, r_range.high,
    # r_count, device=device)  #     spiral_indices = torch.arange(spiral_count, dtype=torch.float32)  #  #  #  #  #
    # two_pi_phi_inverse = 4. * torch.pi / (1. + torch.sqrt(torch.tensor([5.])))  #     thetas = (1. - 2. *  #  #  #
    # spiral_indices / float(spiral_count)).asin()  #     phis = torch.fmod(spiral_indices * two_pi_phi_inverse +  #
    # torch.pi, 2. * torch.pi) - torch.pi  #     rs = rs.repeat(spiral_count, 1)  #     thetas = thetas.unsqueeze(  #
    # -1).repeat(1, r_count)  #     phis = phis.unsqueeze(-1).repeat(1, r_count)  #     return Sinogram3dGrid(phis,
    # thetas, rs)


class OptimisationInstance(ABC):
    @abstractmethod
    def name(self) -> str:
        """
        :return: The name of this algorithm.
        """
        pass

    @abstractmethod
    def step(self) -> bool:
        """
        Execute one step of the optimisation
        :return: Whether the optimisation should terminate.
        """
        pass

    @abstractmethod
    def get_best(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_best_position(self) -> torch.Tensor:
        pass
