import logging

from reg23_app.gui.viewer_singleton import viewer

__all__ = ["InputManager"]

logger = logging.getLogger(__name__)


class InputManager:
    def __init__(self):
        self._key_states = {"Ctrl": False}

        viewer().bind_key("Control", self._on_ctrl_down)

    @property
    def ctrl_pressed(self) -> bool:
        return self._key_states["Ctrl"]

    def _on_ctrl_down(self, _):
        self._key_states["Ctrl"] = True
        yield
        self._key_states["Ctrl"] = False
