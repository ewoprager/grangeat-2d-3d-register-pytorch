from traitlets import HasTraits, Int, Float, Instance, Bool, Enum, Unicode, Undefined, observe, Dict, validate, \
    TraitError, Union

__all__ = ["DRRParams"]


class DRRParams(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    source_distance: float = Float(min=0.0, default_value=1000.0).tag(ui=True)
    x_spacing: float = Float(min=0.0, default_value=0.2).tag(ui=True)
    y_spacing: float = Float(min=0.0, default_value=0.2).tag(ui=True)
    width: int = Int(min=1, default_value=1000).tag(ui=True)
    height: int = Int(min=1, default_value=1000).tag(ui=True)
