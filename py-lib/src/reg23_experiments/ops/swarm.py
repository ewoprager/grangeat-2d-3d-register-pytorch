"""
# `pso`

A simple, flexible particle swarm optimisation package based on `torch.Tensor`s.

A single particle swarm optimisation (PSO) task is performed using an instance of the `Swarm` class. Particles are
initialised on construction of an instance of the `Swarm` class, and iterations are manually executed using the class's
`iterate()` method.

This allows for completely flexible termination conditions and per-iteration behaviour / logic.

An additional element of flexibility is the mutable swarm configuration. The `Swarm`'s configuration is provided and
stored as an instance of the `OptimisationConfig` class. This can be accessed as a property of the `Swarm` instance, and
safely modified between iterations.

## Example use

```python
import torch
from reg23_experiments.ops.swarm import OptimisationConfig, Swarm


def obj_func(xy: torch.Tensor) -> torch.Tensor:
    p = xy[0] - 3.0
    q = xy[1] - 2.0
    return p * p + q * q


config = OptimisationConfig(objective_function=obj_func)
swarm = Swarm(config=config, dimensionality=2, particle_count=20, initialisation_position=torch.zeros(2),
              initialisation_spread=torch.full(2, 10.0), device=torch.device("cpu"))
iteration_count = 20
bests = torch.empty(iteration_count)
for i in range(iteration_count):
    swarm.iterate()
    bests[i] = swarm.current_optimum
...
```
"""

import logging

import traitlets

import torch

__all__ = ["OptimisationConfig", "Swarm"]

logger = logging.getLogger(__name__)


class OptimisationConfig(traitlets.HasTraits):
    """
    A struct that contains configuration information for an instance of the `Swarm` class. These values can safely be
    mutated between iterations of the swarm.
    """
    objective_function = traitlets.Callable()
    inertia_coefficient = traitlets.Float(default_value=0.28)
    cognitive_coefficient = traitlets.Float(default_value=2.525)
    social_coefficient = traitlets.Float(default_value=1.225)


class Swarm:
    """
    An object that stores the state of an ongoing particle swarm optimisation.

    Particles will be initialised upon initialisation of the swarm, which will involve `particle_count` objective
    function evaluations.

    At any point the current optimum value of the objective function found by the swarm can be read using
    Swarm.current_optimum, and the location of that optimum can be read using Swarm.current_optimum_position.

    To perform an iteration of the optimisation, simply call the Swarm `iterate()` method. This will involve
    `particle_count` objective function evaluations.

    The values in the `config` of this class can be read and safely mutated between iterations using `Swarm.config`.
    This allows the objective function and optimisation parameters to be changed on the fly if desired.

    ## Example use
    ```python
    import torch
    from reg23_experiments.ops.swarm import OptimisationConfig, Swarm

    def obj_func(xy: torch.Tensor) -> torch.Tensor:
        p = xy[0] - 3.0
        q = xy[1] - 2.0
        return p * p + q * q

    config = OptimisationConfig(objective_function=obj_func)
    swarm = Swarm(config=config, dimensionality=2, particle_count=20, initialisation_position=torch.zeros(2),
        initialisation_spread=torch.full(2, 10.0), device=torch.device("cpu"))
    iteration_count = 20
    bests = torch.empty(iteration_count)
    for i in range(iteration_count):
        swarm.iterate()
        bests[i] = swarm.current_optimum
    ...
    ```

    ### Implementation details
    A particle is stored as a row of the `particles` tensor:
    | <- position -> | <- velocity -> | <- best position -> | value at best position |
    Let dimensionality be D; the width of the `particles` tensor is therefore 3*D + 1
    """

    def __init__(self, *, config: OptimisationConfig, dimensionality: int, particle_count: int,
                 initialisation_position: torch.Tensor, initialisation_spread: torch.Tensor, device: torch.device,
                 generator: torch.Generator | None = None):
        """
        :param config: The configuration of the swarm. This is stored as a property and can be mutated safely between iterations.
        :param dimensionality: The dimensionality of the search space (the length of the input tensor to the objective function)
        :param particle_count: The number of particles in the swarm.
        :param initialisation_position: The central position in the search space around which the particles will be initialised. One particle will be initialised at exactly this position, for conservation of best found optima between multiple optimisations. This must have size `(dimensionality,)`.
        :param initialisation_spread: The spread of the particles' initialisation positions in the search space around the `initialisation_position`. The value for each dimension is the standard deviation used to sample the particles' initialisation positions in that dimension. This must have size `(dimensionality,)`.
        :param device: The device on which to store all `torch.Tensor`s.
        :param generator: Optional; a generator with which to generate random values for initialisation and movement of the particles.
        """
        self._config = config
        self._dimensionality = dimensionality
        self._generator = generator
        assert initialisation_position.size() == torch.Size([self._dimensionality])
        assert initialisation_spread.size() == torch.Size([self._dimensionality])

        ipo = initialisation_position.to(dtype=torch.float32, device=device)
        isp = initialisation_spread.to(dtype=torch.float32, device=device)
        particle_positions: torch.Tensor = ipo + isp * torch.randn(  #
            [particle_count, dimensionality], dtype=torch.float32, device=device, generator=self._generator)
        # start one particle exactly on the initialisation position
        particle_positions[0] = ipo
        particle_velocities: torch.Tensor = isp * torch.randn(  #
            [particle_count, dimensionality], dtype=torch.float32, device=device, generator=self._generator)
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
        """
        :return: A reference to the swarm's configuration struct. This can safely be mutated between iterations.
        """
        return self._config

    @property
    def dimensionality(self) -> int:
        return self._dimensionality

    @property
    def particle_count(self) -> int:
        return self._particles.size(0)

    @property
    def current_optimum(self) -> torch.Tensor:
        """
        :return: The global best objective function value found so far by the swarm
        """
        return self._global_best

    @property
    def current_optimum_position(self) -> torch.Tensor:
        """
        :return: The position of the global best objective function value found so far by the swarm
        """
        return self._global_best_position

    def iterate(self) -> None:
        """
        Perform one iteration of particle swarm optimisation with the swarm. This involves `particle_count` objective
        function evaluations.
        """
        random_cognitive: torch.Tensor = torch.rand([self.particle_count, 1], dtype=torch.float32, device=self.device,
                                                    generator=self._generator)
        random_social: torch.Tensor = torch.rand([self.particle_count, 1], dtype=torch.float32, device=self.device,
                                                 generator=self._generator)
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
