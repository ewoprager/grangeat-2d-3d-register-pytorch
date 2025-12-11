import os
import traitlets
from traitlets.config import SingletonConfigurable

os.environ["QT_API"] = "PyQt6"

import napari

class ViewerSingleton(SingletonConfigurable):
    _viewer = traitlets.Instance(napari.Viewer, allow_none=True, default_value=None)

    def get(self, **init_kwargs) -> napari.Viewer:
        if self._viewer is None:
            self._viewer = napari.Viewer(**init_kwargs)
            print(f"Viewer initialised with the following parameters: {init_kwargs}")
        return self._viewer

def init_viewer(**kwargs) -> napari.Viewer:
    return ViewerSingleton.instance().get(**kwargs)

def viewer() -> napari.Viewer:
    return ViewerSingleton.instance().get()