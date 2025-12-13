# mock_modules.py
import sys
from unittest.mock import MagicMock

# modules to mock
MOCK_MODULES = [
    "torch", "torch.cuda",
    "napari", "pyswarms",
    "PyQt6", "qtpy.QtWidgets", "matplotlib", "scipy"
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
