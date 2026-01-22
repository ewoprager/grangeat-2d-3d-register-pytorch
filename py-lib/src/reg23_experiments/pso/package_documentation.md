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
from reg23_experiments.pso.swarm import OptimisationConfig, Swarm


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