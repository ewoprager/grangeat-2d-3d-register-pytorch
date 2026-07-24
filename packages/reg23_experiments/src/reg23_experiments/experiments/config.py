import itertools
import logging
from typing import Any, Callable, Generator, Iterable, Iterator, Literal

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

    def __init__(self, value):
        super().__init__(value=value)


class Cartesian(traitlets.HasTraits):
    values: list = traitlets.List(minlen=1)

    def __init__(self, values):
        super().__init__(values=values)


class Range(traitlets.HasTraits):
    range_: list | LinearRange | IntRange | tuple[str, Callable[[float], Any]] = traitlets.Union([  #
        traitlets.List(minlen=2),  #
        traitlets.Instance(LinearRange, allow_none=False),  #
        traitlets.Instance(IntRange, allow_none=False),  #
        traitlets.Tuple(traitlets.Unicode(allow_none=False), traitlets.Callable()),  #
    ])

    def __init__(self, range_):
        super().__init__(range_=range_)

    def sample(self, float01: float) -> Any:
        if isinstance(self.range_, list):
            return self.range_[min(len(self.range_) - 1, int(np.floor(float01 * float(len(self.range_)))))]
        elif isinstance(self.range_, LinearRange):
            return self.range_.low + float01 * (self.range_.high - self.range_.low)
        elif isinstance(self.range_, IntRange):
            width = self.range_.high - self.range_.low
            return self.range_.low + min(width, int(np.floor(float01 * float(width + 1))))
        else:  # self.range_ is a named function
            return self.range_[1](float01)

    def serialize(self) -> dict[str, Any] | list:
        if isinstance(self.range_, list):
            return self.range_
        elif isinstance(self.range_, LinearRange):
            return {  #
                "range_type": "LinearRange",  #
                "low": self.range_.low,  #
                "high": self.range_.high,  #
            }
        elif isinstance(self.range_, IntRange):
            return {  #
                "range_type": "IntRange",  #
                "low": self.range_.low,  #
                "high": self.range_.high,  #
            }
        else:  # self.range_ is a named function
            return {  #
                "range_type": "function",  #
                "name": self.range_[0],  #
            }


class _Configs(Iterable[tuple[str, dict[str, Any]]]):
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
        ordered_cartesian_names: list[str] = [  #
            key  #
            for key, value in values.items()  #
            if isinstance(value, Cartesian)  #
        ]
        if ordered_cartesian_names:
            indexed_cartesian_grid: Iterator = itertools.product(*[  #
                enumerate(values[name].values) for name in ordered_cartesian_names  #
            ])
            cart_n = int(np.prod([len(value.values) for _, value in values.items() if isinstance(value, Cartesian)]))
            self._cart_generator: Callable[[], Generator[tuple[str, dict[str, Any]]]] | None = lambda: (  #
                (  #
                    "c" + "-".join(f"{i}" for i, _ in indexed_cart_values),  #
                    {  #
                        name: value  #
                        for name, (_, value) in zip(ordered_cartesian_names, indexed_cart_values)  #
                    },  #
                )  #
                for indexed_cart_values in indexed_cartesian_grid  #
            )
        else:
            cart_n = 1
            self._cart_generator: Callable[[], Generator[tuple[str, dict[str, Any]]]] | None = None
        # -----
        # Spatial sampling of values
        range_samplers: dict[str, Callable[[float], Any]] = {  #
            key: value.sample  #
            for key, value in values.items()  #
            if isinstance(value, Range)  #
        }
        if range_samplers:
            if space_sample_count is None or space_sample_count < 1:
                space_sample_count: int = 2 ** len(range_samplers)
                logger.info(
                    f"No/invalid space sample count specified for config iteration; defaulting to 2^n_variables, "
                    f"= {space_sample_count}")
            if space_sample_method == "sobol" and (space_sample_count & (space_sample_count - 1)) != 0:
                space_sample_count = 2 ** int(np.round(np.log2(float(space_sample_count))))
                logger.warning(
                    f"The space sample count for Sobol sampling must be a power of 2. Rounding to the nearest power "
                    f"of 2, = {space_sample_count}")
            # ToDo: Currently only do Sobol
            sobol_samples = scipy.stats.qmc.Sobol(len(range_samplers)).random_base2(  #
                space_sample_count.bit_length() - 1  #
            )  # size (space_sample_count, len(range_samplers))
            self._range_generator: Callable[[], Generator[tuple[str, dict[str, Any]]]] | None = lambda: (  #
                (  #
                    f"s{i}",  #
                    {  #
                        name: sampler(float01)  #
                        for float01, (name, sampler) in zip(float01s, range_samplers.items())  #
                    }  #
                )  #
                for i, float01s in enumerate(sobol_samples)  #
            )
        else:
            space_sample_count = 1
            self._range_generator: Callable[[], Generator[tuple[str, dict[str, Any]]]] | None = None
        # -----
        # Total number of configs
        self._len = cart_n * space_sample_count

    def __iter__(self) -> Iterator[tuple[str, dict[str, Any]]]:
        if self._cart_generator is None:
            if self._range_generator is None:
                yield "only_config", self._constants
            else:
                for range_name, range_config in self._range_generator():
                    yield range_name, range_config | self._constants
        else:
            if self._range_generator is None:
                for cart_name, cart_config in self._cart_generator():
                    yield cart_name, cart_config | self._constants
            else:
                for cart_name, cart_config in self._cart_generator():
                    intermediate = cart_config | self._constants
                    for range_name, range_config in self._range_generator():
                        yield cart_name + "_" + range_name, range_config | intermediate

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

    def __init__(self, values):
        super().__init__(values=values)

    def iterable(  #
            self,  #
            *,  #
            space_sample_method: Literal["sobol"] = "sobol",  #
            space_sample_count: int | None = None,  #
    ) -> _Configs:
        return _Configs(  #
            self.values,  #
            space_sample_method=space_sample_method,  #
            space_sample_count=space_sample_count,  #
        )

    def serialize(self) -> dict[str, Any]:
        return {  #
            "constants": {  #
                key: value.value  #
                for key, value in self.values.items()  #
                if isinstance(value, Constant)  #
            },  #
            "cartesian": {  #
                key: value.values  #
                for key, value in self.values.items()  #
                if isinstance(value, Cartesian)  #
            },  #
            "range": {  #
                key: value.serialize()  #
                for key, value in self.values.items()  #
                if isinstance(value, Range)  #
            },  #
        }
