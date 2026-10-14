import logging
import os

os.environ["QT_API"] = "PyQt6"

from reg23_app.context import AppContext
from reg23_app.gui.widgets.hastraits_widget import HasTraitsWidget

__all__ = ["ParametersWidget"]

logger = logging.getLogger(__name__)


class ParametersWidget(HasTraitsWidget):
    def __init__(self, ctx: AppContext):
        super().__init__(ctx.state.parameters)
