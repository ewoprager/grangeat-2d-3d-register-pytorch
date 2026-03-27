from traitlets import Float, HasTraits

__all__ = ["GUISettings"]


class GUISettings(HasTraits):
    """
    Only contains data; either simple values, or other `HasTraits` instances that themselves just contain data.
    """
    rotation_sensitivity: float = Float(min=0.0005, max=0.05, default_value=0.002).tag(ui=True)
    translation_sensitivity: float = Float(min=0.005, max=0.5, default_value=0.06).tag(ui=True)
