import torch
from jaxtyping import Float64
from traitlets import traitlets

__all__ = ["Volume", "SeriesDescription", "OneSeries"]


class Volume(traitlets.HasTraits):
    uid: str = traitlets.Unicode(allow_none=False)
    raw_data: torch.Tensor = traitlets.Instance(torch.Tensor, allow_none=False)
    rescale_slope: float = traitlets.Float(allow_none=False, default_value=1.0)
    rescale_intercept: float = traitlets.Float(allow_none=False, default_value=0.0)
    rescale_type: str | None = traitlets.Unicode(allow_none=True, default_value=None)
    spacing: Float64[torch.Tensor, "3"] = traitlets.Instance(torch.Tensor, allow_none=False)
    image_position_patient: Float64[torch.Tensor, "3"] | None = traitlets.Instance(torch.Tensor, allow_none=True,
                                                                                   default_value=None)


class OneSeries(traitlets.HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    file_type: str = traitlets.Unicode(allow_none=False)


class SeriesDescription(traitlets.HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    file_type: str = traitlets.Unicode(allow_none=False)
    uid: str = traitlets.Unicode(allow_none=False)
    slice_count: int = traitlets.Int(allow_none=False)
    number: int | None = traitlets.Integer(allow_none=True)
    description: str | None = traitlets.Unicode(allow_none=True)
    protocol_name: str | None = traitlets.Unicode(allow_none=True)
