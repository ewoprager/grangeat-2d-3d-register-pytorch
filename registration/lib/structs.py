from typing import NamedTuple, Tuple

import torch
import kornia
import scipy
import numpy


class LinearMapping:
    def __init__(self, intercept: float | torch.Tensor, gradient: float | torch.Tensor):
        self.intercept = intercept
        self.gradient = gradient

    def __call__(self, x: float | torch.Tensor) -> float | torch.Tensor:
        return self.intercept + self.gradient * x


class LinearRange:
    def __init__(self, low: float, high: float, ):
        self.low = low
        self.high = high

    def generate_grid(self, count: int, *, device=torch.device('cpu')) -> torch.Tensor:
        return torch.linspace(self.low, self.high, count, device=device)

    def get_mapping_from(self, other: 'LinearRange') -> LinearMapping:
        frac: float = (self.high - self.low) / (other.high - other.low)
        return LinearMapping(self.low - frac * other.low, frac)

    def get_spacing(self, count: int) -> float:
        return (self.high - self.low) / float(count - 1)

    def get_centre(self) -> float:
        return .5 * (self.low + self.high)

    @classmethod
    def grid_sample_range(cls):
        return LinearRange(-1., 1.)


class Transformation(NamedTuple):
    rotation: torch.Tensor
    translation: torch.Tensor

    def inverse(self) -> 'Transformation':
        r_inverse = kornia.geometry.conversions.axis_angle_to_rotation_matrix(-self.rotation.unsqueeze(0))[0]
        r_inverse_t = torch.einsum('kl,...l->...k', r_inverse, self.translation.unsqueeze(0))[0]
        return Transformation(-self.rotation, -r_inverse_t)

    def get_h(self, *, device=torch.device('cpu')) -> torch.Tensor:
        """
        :param device:
        :return: [(4, 4) tensor] The homogenous affine transformation matrix H corresponding to this transformation
        """
        r = kornia.geometry.conversions.axis_angle_to_rotation_matrix(self.rotation.unsqueeze(0))[0].to(device=device,
                                                                                                        dtype=torch.float32)
        rt = torch.hstack([r, self.translation.to(device=device).t().unsqueeze(-1)])
        return torch.vstack([rt, torch.tensor([0., 0., 0., 1.], device=device).unsqueeze(0)])

    def vectorised(self) -> torch.Tensor:
        return torch.cat((self.rotation, self.translation), dim=0)

    def to(self, **kwargs) -> 'Transformation':
        return Transformation(self.rotation.to(**kwargs), self.translation.to(**kwargs))

    def device_consistent(self) -> bool:
        return self.rotation.device == self.translation.device

    def distance(self, other: 'Transformation') -> float:
        device = self.translation.device
        r1 = kornia.geometry.conversions.axis_angle_to_rotation_matrix(self.rotation.unsqueeze(0))[0].to(device=device,
                                                                                                         dtype=torch.float32)
        r2 = kornia.geometry.conversions.axis_angle_to_rotation_matrix(other.rotation.unsqueeze(0))[0].to(device=device,
                                                                                                          dtype=torch.float32)
        return (((self.translation - other.translation) / 100.).square().sum() + torch.tensor(
            [numpy.real(scipy.linalg.logm((torch.matmul(r1.t(), r2).cpu().numpy())))],
            device=device).square().sum()).sqrt().item()

    @classmethod
    def zero(cls, *, device=torch.device('cpu')) -> 'Transformation':
        return Transformation(torch.zeros(3, device=device), torch.zeros(3, device=device))

    @classmethod
    def random(cls, *, device=torch.device('cpu')) -> 'Transformation':
        return Transformation(torch.pi * (-1. + 2. * torch.rand(3, device=device)),
                              25. * (-1. + 2. * torch.rand(3, device=device)) + Transformation.zero(
                                  device=device).translation)


class SceneGeometry(NamedTuple):
    source_distance: float  # [mm]; distance in the positive z-direction from the centre of the detector array

    def source_position(self, *, device=torch.device('cpu')):
        return torch.tensor([0., 0., self.source_distance], device=device)

    @classmethod
    def projection_matrix(cls, source_position: torch.Tensor, central_ray: torch.Tensor | None = None) -> torch.Tensor:
        """

        :param source_position: [(3,) tensor] the position of the X-ray source
        :param central_ray: [(3,) tensor or None] the vector from the X-ray source to the closest point on the
        detector array. If none, the detector array is assumed to be the x-y plane.
        :return: [(4, 4) tensor] The projection matrix P that projects points in homogeneous coordinates away from
        the given source position onto the plane of the detector array, as given by the central ray.
        """
        device = source_position.device

        if central_ray is None:
            central_ray = torch.tensor([0., 0., - source_position[2]], device=device)

        assert central_ray.device == device
        assert source_position.size() == torch.Size([3])
        assert central_ray.size() == torch.Size([3])

        m_matrix: torch.Tensor = torch.outer(torch.hstack((source_position, torch.tensor([1.], device=device))),
                                             central_ray) + torch.dot(central_ray, central_ray) * torch.vstack(
            (torch.eye(3, device=device), torch.zeros((1, 3), device=device)))

        return torch.hstack((m_matrix, -torch.matmul(m_matrix, source_position.t().unsqueeze(-1))))


class Sinogram2dRange(NamedTuple):
    phi: LinearRange
    r: LinearRange


class Sinogram3dRange(NamedTuple):
    phi: LinearRange
    theta: LinearRange
    r: LinearRange

    def get_spacing(self, counts: int | Tuple[int, int, int] | torch.Size, *,
                    device=torch.device('cpu')) -> torch.Tensor:
        if isinstance(counts, int):
            counts = (counts, counts, counts)
        elif isinstance(counts, torch.Size):
            assert len(counts) == 3
        return torch.tensor(
            [self.r.get_spacing(counts[0]), self.theta.get_spacing(counts[1]), self.phi.get_spacing(counts[2])],
            device=device)

    def get_centres(self, *, device=torch.device('cpu')) -> torch.Tensor:
        return torch.tensor([self.r.get_centre(), self.theta.get_centre(), self.phi.get_centre()], device=device)


class Sinogram2dGrid(NamedTuple):
    phi: torch.Tensor
    r: torch.Tensor

    def to(self, **kwargs) -> 'Sinogram2dGrid':
        return Sinogram2dGrid(self.phi.to(**kwargs), self.r.to(**kwargs))

    def device_consistent(self) -> bool:
        return self.phi.device == self.r.device

    def size_consistent(self) -> bool:
        return self.phi.size() == self.r.size()

    @classmethod
    def linear_from_range(cls, sinogram_range: Sinogram2dRange, counts: int | Tuple[int, int] | torch.Size, *,
                          device=torch.device("cpu")) -> 'Sinogram2dGrid':
        if isinstance(counts, int):
            counts = (counts, counts)
        phis = torch.linspace(sinogram_range.phi.low, sinogram_range.phi.high, counts[0], device=device)
        rs = torch.linspace(sinogram_range.r.low, sinogram_range.r.high, counts[1], device=device)
        phis, rs = torch.meshgrid(phis, rs)
        return Sinogram2dGrid(phis, rs)


class Sinogram3dGrid(NamedTuple):
    phi: torch.Tensor
    theta: torch.Tensor
    r: torch.Tensor

    def to(self, **kwargs) -> 'Sinogram3dGrid':
        return Sinogram3dGrid(self.phi.to(**kwargs), self.theta.to(**kwargs), self.r.to(**kwargs))

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

    @classmethod
    def linear_from_range(cls, sinogram_range: Sinogram3dRange, counts: int | Tuple[int, int, int] | torch.Size, *,
                          device=torch.device("cpu")) -> 'Sinogram3dGrid':
        if isinstance(counts, int):
            counts = (counts, counts, counts)
        elif isinstance(counts, torch.Size):
            assert len(counts) == 3
        phis = torch.linspace(sinogram_range.phi.low, sinogram_range.phi.high, counts[0], device=device)
        thetas = torch.linspace(sinogram_range.theta.low, sinogram_range.theta.high, counts[1], device=device)
        rs = torch.linspace(sinogram_range.r.low, sinogram_range.r.high, counts[2], device=device)
        phis, thetas, rs = torch.meshgrid(phis, thetas, rs)
        return Sinogram3dGrid(phis, thetas, rs)

    @classmethod
    def fibonacci_from_r_range(cls, r_range: LinearRange, r_count: int, *, spiral_count: int | None = None,
                               device=torch.device("cpu")) -> 'Sinogram3dGrid':
        if spiral_count is None:
            spiral_count = r_count * r_count
        rs = torch.linspace(r_range.low, r_range.high, r_count, device=device)
        spiral_indices = torch.arange(spiral_count, dtype=torch.float32)
        two_pi_phi_inverse = 4. * torch.pi / (1. + torch.sqrt(torch.tensor([5.])))
        thetas = (1. - 2. * spiral_indices / float(spiral_count)).asin()
        phis = torch.fmod(spiral_indices * two_pi_phi_inverse + torch.pi, 2. * torch.pi) - torch.pi
        rs = rs.repeat(spiral_count, 1)
        thetas = thetas.unsqueeze(-1).repeat(1, r_count)
        phis = phis.unsqueeze(-1).repeat(1, r_count)
        return Sinogram3dGrid(phis, thetas, rs)
