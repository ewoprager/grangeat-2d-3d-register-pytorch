import logging

import traitlets

import torch

__all__ = ["OptimisationConfig", "Swarm"]

logger = logging.getLogger(__name__)


class OptimisationConfig(traitlets.HasTraits):
    objective_function = traitlets.Callable()
    inertia_coefficient = traitlets.Float(default_value=0.28)
    cognitive_coefficient = traitlets.Float(default_value=2.525)
    social_coefficient = traitlets.Float(default_value=1.225)


class Swarm:
    """
    A particle is stored as a row of the `particles` tensor:
    | <- position -> | <- velocity -> | <- best position -> | value at best position |
    Let dimensionality be D; the width of the `particles` tensor is therefore 3*D + 1
    """

    def __init__(self, *, config: OptimisationConfig, dimensionality: int, particle_count: int,
                 initialisation_position: torch.Tensor, initialisation_spread: torch.Tensor, device: torch.device,
                 generator: torch.Generator | None = None):
        self._config = config
        self._dimensionality = dimensionality
        assert initialisation_position.size() == torch.Size([self._dimensionality])
        assert initialisation_spread.size() == torch.Size([self._dimensionality])

        ipo = initialisation_position.to(dtype=torch.float32, device=device)
        isp = initialisation_spread.to(dtype=torch.float32, device=device)
        particle_positions: torch.Tensor = ipo + isp * torch.rand(  #
            [particle_count, dimensionality], dtype=torch.float32, device=device, generator=generator)
        # start one particle exactly on the initialisation position
        particle_positions[0] = ipo
        particle_velocities: torch.Tensor = isp * torch.randn(  #
            [particle_count, dimensionality], dtype=torch.float32, device=device, generator=generator)
        self._particles = torch.cat([particle_positions, particle_velocities, particle_positions,
                                     torch.zeros([particle_count, 1], dtype=torch.float32, device=device)], dim=1)
        # evaluating for the first particle
        self._particles[0, -1] = self._config.objective_function(self._particles[0, 0:dimensionality])
        # initialising to determine global best
        self._global_best_position = self._particles[0, 0:dimensionality]
        self._global_best = self._particles[0, -1]
        # evaluating for rest of particles, and determining global best
        for particle in range(particle_count):
            self._particles[particle, -1] = self._config.objective_function(self._particles[particle, 0:dimensionality])
            if self._particles[particle, -1] < self._global_best:
                self._global_best = self._particles[particle, -1]
                self._global_best_position = self._particles[particle, 0:dimensionality]

    @property
    def device(self) -> torch.device:
        return self._particles.device

    @property
    def config(self) -> OptimisationConfig:
        return self._config

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def particle_count(self) -> int:
        return self._particles.size(0)

    @property
    def current_optimum(self) -> torch.Tensor:
        return self._global_best

    @property
    def current_optimum_position(self) -> torch.Tensor:
        return self._global_best_position

    def iterate(self) -> None:
        random_cognitive: torch.Tensor = torch.rand([self.particle_count, 1], dtype=torch.float32, device=self.device)
        random_social: torch.Tensor = torch.rand([self.particle_count, 1], dtype=torch.float32, device=self.device)
        # updating velocities
        self._particles[:, self.dimensionality:2 * self.dimensionality] = (  #
                self.config.inertia_coefficient * self._particles[:, self.dimensionality:2 * self.dimensionality] +  #
                self.config.cognitive_coefficient * random_cognitive * (  #
                        self._particles[:, 2 * self.dimensionality:3 * self.dimensionality] -  #
                        self._particles[:, 0:self.dimensionality]) +  #
                self.config.social_coefficient * random_social * (  #
                        self._global_best_position.unsqueeze(0) -  #
                        self._particles[:, 0:self.dimensionality]))
        # updating positions
        self._particles[:, 0:self.dimensionality] = (  #
                self._particles[:, 0:self.dimensionality] +  #
                self._particles[:, self.dimensionality:2 * self.dimensionality])
        # evaluating objective function
        for particle in range(self.particle_count):
            objective_function_value = self.config.objective_function(self._particles[particle, 0:self.dimensionality])
            if objective_function_value < self._particles[particle, -1]:
                self._particles[particle, -1] = objective_function_value
                self._particles[particle, 2 * self.dimensionality:3 * self.dimensionality] = self._particles[
                    particle, 0:self.dimensionality]
                if objective_function_value < self._global_best:
                    self._global_best = objective_function_value
                    self._global_best_position = self._particles[particle, 0:self.dimensionality]
