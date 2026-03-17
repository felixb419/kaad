import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "kaad"

extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.mathjax",
]

# Breathe config (connects to Doxygen XML)
breathe_projects = {"kaad": "../docs/doxygen/xml"}
breathe_default_project = "kaad"

# Markdown support
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

html_theme = "sphinx_rtd_theme"
