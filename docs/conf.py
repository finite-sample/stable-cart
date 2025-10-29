"""Configuration for Sphinx documentation build."""

import pathlib
import sys

# -- Path setup --------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -- Project information -----------------------------------------------------

project = "stable-cart"
author = "stable-cart developers"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "numpy",
    "sklearn",
    "sklearn.base",
    "sklearn.preprocessing",
    "sklearn.linear_model",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_static_path = ["_static"]
