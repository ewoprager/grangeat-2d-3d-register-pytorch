import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # make project importable

project = "reg23-experiments"
author = "Edmund Prager"
extensions = [  #
    "sphinx.ext.autodoc",  #
    "sphinx.ext.autosummary",  #
    # "sphinx.ext.napoleon",  # Google/NumPy docstring support
    "sphinx.ext.viewcode",  # add source links
    "myst_parser"  #
]
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"  # or "sphinx_rtd_theme" etc.
html_static_path = ["_static"]
