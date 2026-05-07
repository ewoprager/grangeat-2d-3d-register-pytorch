from typing import Any, Generator, NamedTuple

import torch

__all__ = ["TensorPolicy"]


class TensorPolicy(NamedTuple):
    device: torch.device
    dtype: torch.dtype

    @property
    def as_kwargs(self) -> dict[str, Any]:
        return self._asdict()

    def apply(self, *args: torch.Tensor) -> Generator[torch.Tensor, None, None]:
        return (  #
            tensor.to(**self.as_kwargs)  #
            for tensor in args  #
        )

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> 'TensorPolicy':
        return TensorPolicy(device=tensor.device, dtype=tensor.dtype)
