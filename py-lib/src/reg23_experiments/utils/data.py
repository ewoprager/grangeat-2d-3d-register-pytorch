import traitlets

__all__ = ["StrictHasTraits"]


class StrictHasTraits(traitlets.HasTraits):
    """
    A `HasTraits` subclass with strict `__init__` semantics:
      - unknown kwargs are rejected
      - traits with `default_value=Undefined` are required
    """

    def __init__(self, **kwargs):
        # reject unknown kwargs
        valid_traits = set(self.traits())
        unknown = set(kwargs) - valid_traits
        if unknown:
            cls = type(self).__name__
            raise TypeError(f"{cls} got unexpected keyword arguments: {unknown}")

        super().__init__(**kwargs)

        # enforce required traits
        missing = {  #
            name for name, trait in self.traits().items() if
            trait.default_value is traitlets.Undefined and getattr(self, name) is traitlets.Undefined  #
        }

        if missing:
            cls = type(self).__name__
            raise TypeError(f"{cls} missing required arguments: {missing}")
