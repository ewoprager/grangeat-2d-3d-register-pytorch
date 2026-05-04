import logging

import napari.layers
import torch

from reg23_app.context import AppContext
from reg23_app.gui.viewer_singleton import viewer
from reg23_experiments.data.structs import Error


__all__ = ["add_slice_view_layer"]

logger = logging.getLogger(__name__)

class _SliceViewLayerManager:
    def __init__(self):
        pass

def add_slice_view_layer(*, ctx: AppContext) -> None:
    pass