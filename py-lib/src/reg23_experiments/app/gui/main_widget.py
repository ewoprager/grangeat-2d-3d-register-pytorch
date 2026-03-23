import os
import logging

os.environ["QT_API"] = "PyQt6"

import napari
from magicgui import widgets

from reg23_experiments.app.context import AppContext

__all__ = ["MainWidget"]

logger = logging.getLogger(__name__)

class MainWidget(widgets.Container):
    def __init__(self, ctx: AppContext):
        super().__init__(labels=True)
        self._ctx = ctx

