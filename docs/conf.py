"""Configuration for Sphinx documentation build."""

import pathlib
import sys

# -- Path setup --------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# -- Project information -----------------------------------------------------

project = "stable-cart"
author = "stable-cart developers"
copyright = "2024, stable-cart developers"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "special-members": "__init__",
    "undoc-members": True,
}

autodoc_mock_imports = [
    "numpy",
    "sklearn",
    "sklearn.base",
    "sklearn.preprocessing",
    "sklearn.linear_model",
    "sklearn.tree",
    "sklearn.utils",
    "sklearn.utils.validation",
    "sklearn.metrics",
    "sklearn.model_selection",
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"{project} v{release}"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#2980b9",
        "color-brand-content": "#2980b9",
        "color-api-name": "#2980b9",
        "color-api-pre-name": "#6c757d",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3498db",
        "color-brand-content": "#3498db",
        "color-api-name": "#3498db",
        "color-api-pre-name": "#adb5bd",
    },
    "announcement": "<strong>🚀 Stable-CART:</strong> Variance-reduced tree ensembles for more stable predictions.",
}

# Add custom CSS
html_css_files = [
    "custom.css",
]
