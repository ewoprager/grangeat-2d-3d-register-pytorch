import itertools
import logging
from typing import Any, Callable, Iterable, Iterator, Literal

import numpy as np
import scipy
import traitlets

from reg23_experiments.data.structs import LinearRange

__all__ = ["IntRange", "Constant", "Cartesian", "Range", "ExperimentConfig"]

logger = logging.getLogger(__name__)


class IntRange:
    """
    The range (low, high) is __inclusive__.
    """

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high


class Constant(traitlets.HasTraits):
    value: Any = traitlets.Any(default_value=traitlets.Undefined)


class Cartesian(traitlets.HasTraits):
    values: list = traitlets.List(minlen=1)


class Range(traitlets.HasTraits):
    range: list | LinearRange | IntRange = traitlets.Union([  #
        traitlets.List(minlen=2),  #
        traitlets.Instance(LinearRange, allow_none=False),  #
        traitlets.Instance(IntRange, allow_none=False),  #
    ])

    def sample(self, float01: float) -> Any:
        if isinstance(self.range, list):
            return self.range[min(len(self.range) - 1, int(np.floor(float01 * float(len(self.range)))))]
        elif isinstance(self.range, LinearRange):
            return self.range.low + float01 * (self.range.high - self.range.low)
        else:  # self.range is an IntRange:
            width = self.range.high - self.range.low
            return self.range.low + min(width, int(np.floor(float01 * float(width + 1))))


class _Configs(Iterable[dict[str, Any]]):
    def __init__(  #
            self,  #
            values: dict[str, Constant | Cartesian | Range],  #
            space_sample_method: Literal["sobol"] = "sobol",  #
            space_sample_count: int | None = None,  #
    ):
        # -----
        # Constant values
        self._constants: dict[str, Any] = {  #
            key: value.value  #
            for key, value in values.items()  #
            if isinstance(value, Constant)  #
        }
        # -----
        # Cartesian grid of values
        self._ordered_cartesian_names: list[str] = [  #
            key  #
            for key, value in values.items()  #
            if isinstance(value, Cartesian)  #
        ]
        self._cartesian_grid: Iterator = itertools.product(*[  #
            values[name].values for name in self._ordered_cartesian_names  #
        ])
        cart_n = int(np.prod([len(value.values) for _, value in values.items() if isinstance(value, Cartesian)]))
        # -----
        # Spatial sampling of values
        self._space_sample_method = space_sample_method
        self._range_samplers: dict[str, Callable[[float], Any]] = {  #
            key: value.sample  #
            for key, value in values.items()  #
            if isinstance(value, Range)  #
        }
        if self._range_samplers:
            if space_sample_count is None or space_sample_count < 1:
                self._space_sample_count: int = 2 ** len(self._range_samplers)
                logger.info(
                    f"No/invalid space sample count specified for config iteration; defaulting to 2^n_variables, "
                    f"= {self._space_sample_count}")
            else:
                self._space_sample_count: int = space_sample_count
            if self._space_sample_method == "sobol" and (
                    self._space_sample_count & (self._space_sample_count - 1)) != 0:
                self._space_sample_count = 2 ** int(np.round(np.log2(float(self._space_sample_count))))
                logger.warning(
                    f"The space sample count for Sobol sampling must be a power of 2. Rounding to the nearest power "
                    f"of 2, = {self._space_sample_count}")
        else:
            self._space_sample_count = 1
        # -----
        # Total number of configs
        self._len = cart_n * self._space_sample_count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._ordered_cartesian_names:
            for cart_values in self._cartesian_grid:
                cart_instance = {  #
                                    name: value  #
                                    for name, value in zip(self._ordered_cartesian_names, cart_values)  #
                                } | self._constants
                if self._range_samplers:
                    # ToDo: Currently only do Sobol
                    sobol_sequence = scipy.stats.qmc.Sobol(len(self._range_samplers))
                    all_float01s: np.ndarray = sobol_sequence.random_base2(  #
                        self._space_sample_count.bit_length() - 1  #
                    )  # size (self._space_sample_count, len(self._range_samplers))
                    for float01s in all_float01s:
                        yield {  #
                            name: sampler(float01)  #
                            for float01, (name, sampler) in zip(float01s, self._range_samplers.items())  #
                        } | cart_instance
                else:
                    yield cart_instance
        else:
            if self._range_samplers:
                # ToDo: Currently only do Sobol
                sobol_sequence = scipy.stats.qmc.Sobol(len(self._range_samplers))
                all_float01s: np.ndarray = sobol_sequence.random_base2(  #
                    self._space_sample_count.bit_length() - 1  #
                )  # size (self._space_sample_count, len(self._range_samplers))
                for float01s in all_float01s:
                    yield {  #
                        name: sampler(float01)  #
                        for float01, (name, sampler) in zip(float01s, self._range_samplers.items())  #
                    } | self._constants
            else:
                yield self._constants

    def __len__(self) -> int:
        return self._len


class ExperimentConfig(traitlets.HasTraits):
    values: dict[str, Constant | Cartesian | Range] = traitlets.Dict(  #
        key_trait=traitlets.Unicode(allow_none=False),  #
        value_trait=traitlets.Union([  #
            traitlets.Instance(Constant, allow_none=False),  #
            traitlets.Instance(Cartesian, allow_none=False),  #
            traitlets.Instance(Range, allow_none=False),  #
        ]),  #
    )

    def iterable(  #
            self,  #
            space_sample_method: Literal["sobol"] = "sobol",  #
            space_sample_count: int | None = None,  #
    ) -> _Configs:
        return _Configs(  #
            self.values,  #
            space_sample_method=space_sample_method,  #
            space_sample_count=space_sample_count,  #
        )
