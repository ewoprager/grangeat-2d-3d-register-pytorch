import os
import sys

sys.path.insert(0, os.path.abspath("../"))  # make project importable

project = "reg23-experiments"
author = "Edmund Prager"
extensions = [  #
    "sphinx.ext.autodoc",  #
    "sphinx.ext.autosummary",  #
    "sphinx.ext.napoleon",  # Google/NumPy docstring support
    "sphinx.ext.viewcode",  # add source links
    "myst_parser"  #
]
autosummary_generate = True

autodoc_mock_imports = ["torch", "reg23", "PyQt6", "numpy", "torch", "matplotlib", ]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"  # or "alabaster" etc.
html_static_path = ["_static"]
