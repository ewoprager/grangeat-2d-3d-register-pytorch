import torch
import traitlets
from jaxtyping import Float64

__all__ = ["OrderedPoints2D", "NamedPoints2D"]


class OrderedPoints2D(traitlets.HasTraits):
    data: Float64[torch.Tensor, "n 2"] = traitlets.Instance(  #
        torch.Tensor,  #
        allow_none=False,  #
        default_value=torch.empty((0, 2), dtype=torch.float64)  #
    )

    @property
    def count(self) -> int:
        return self.data.size()[0]

    @traitlets.validate("data")
    def _validate_data(self, proposal):
        if proposal["value"].dtype != torch.float64:
            raise traitlets.TraitError("OrderedPoints2D must contain a tensor of torch.float64s")
        if len(proposal["value"].size()) != 2:
            raise traitlets.TraitError("OrderedPoints2D must contain a tensor of size (N, 2)")
        if proposal["value"].size()[1] != 2:
            raise traitlets.TraitError("OrderedPoints2D must contain a tensor of size (N, 2)")
        return proposal["value"]


class NamedPoints2D(traitlets.HasTraits):
    names: list[str] = traitlets.List(trait=traitlets.Unicode(allow_none=False))
    data: Float64[torch.Tensor, "n 2"] = traitlets.Instance(  #
        torch.Tensor,  #
        allow_none=False,  #
        default_value=torch.empty((0, 2), dtype=torch.float64)  #
    )

    @traitlets.validate("data")
    def _validate_data(self, proposal):
        if proposal["value"].dtype != torch.float64:
            raise traitlets.TraitError("NamedPoints2D must contain a tensor of torch.float64s")
        if len(proposal["value"].size()) != 2:
            raise traitlets.TraitError("NamedPoints2D must contain a tensor of size (N, 2)")
        if proposal["value"].size()[1] != 2:
            raise traitlets.TraitError("NamedPoints2D must contain a tensor of size (N, 2)")
        return proposal["value"]
