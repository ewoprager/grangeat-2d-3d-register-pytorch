import torch
import traitlets
from jaxtyping import Float64

__all__ = ["OrderedPoints2D", "OrderedPoints3D", "NamedPoints2D", "NamedPoints3D"]


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


class OrderedPoints3D(traitlets.HasTraits):
    data: Float64[torch.Tensor, "n 3"] = traitlets.Instance(  #
        torch.Tensor,  #
        allow_none=False,  #
        default_value=torch.empty((0, 3), dtype=torch.float64)  #
    )

    @property
    def count(self) -> int:
        return self.data.size()[0]

    @traitlets.validate("data")
    def _validate_data(self, proposal):
        if proposal["value"].dtype != torch.float64:
            raise traitlets.TraitError("OrderedPoints3D must contain a tensor of torch.float64s")
        if len(proposal["value"].size()) != 2:
            raise traitlets.TraitError("OrderedPoints3D must contain a tensor of size (N, 3)")
        if proposal["value"].size()[1] != 3:
            raise traitlets.TraitError("OrderedPoints3D must contain a tensor of size (N, 3)")
        return proposal["value"]


class NamedPoints2D(traitlets.HasTraits):
    names: list[str] = traitlets.List(trait=traitlets.Unicode(allow_none=False))
    data: Float64[torch.Tensor, "n 2"] = traitlets.Instance(  #
        torch.Tensor,  #
        allow_none=False,  #
        default_value=torch.empty((0, 2), dtype=torch.float64)  #
    )

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        return zip(self.names, self.data)

    @traitlets.validate("data")
    def _validate_data(self, proposal):
        if proposal["value"].dtype != torch.float64:
            raise traitlets.TraitError("NamedPoints2D must contain a tensor of torch.float64s")
        if len(proposal["value"].size()) != 2:
            raise traitlets.TraitError("NamedPoints2D must contain a tensor of size (N, 2)")
        if proposal["value"].size()[1] != 2:
            raise traitlets.TraitError("NamedPoints2D must contain a tensor of size (N, 2)")
        return proposal["value"]


class NamedPoints3D(traitlets.HasTraits):
    names: list[str] = traitlets.List(trait=traitlets.Unicode(allow_none=False))
    data: Float64[torch.Tensor, "n 3"] = traitlets.Instance(  #
        torch.Tensor,  #
        allow_none=False,  #
        default_value=torch.empty((0, 3), dtype=torch.float64)  #
    )

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        return zip(self.names, self.data)

    @traitlets.validate("data")
    def _validate_data(self, proposal):
        if proposal["value"].dtype != torch.float64:
            raise traitlets.TraitError("NamedPoints3D must contain a tensor of torch.float64s")
        if len(proposal["value"].size()) != 2:
            raise traitlets.TraitError("NamedPoints3D must contain a tensor of size (N, 3)")
        if proposal["value"].size()[1] != 3:
            raise traitlets.TraitError("NamedPoints3D must contain a tensor of size (N, 3)")
        return proposal["value"]
