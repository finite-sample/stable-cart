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

# Get version from package
try:
    import stable_cart
    release = stable_cart.__version__
except ImportError:
    # Fallback for docs builds where package isn't installed
    try:
        from importlib.metadata import version
        release = version("stable-cart")
    except Exception:
        release = "1.0.0"  # Fallback version

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

autosummary_generate = False

# Remove module names from autosummary  
add_module_names = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "special-members": "__init__",
    "undoc-members": False,  # Set to False to avoid clutter
    "inherited-members": True,
    "exclude-members": "__dict__,__weakref__,__module__",
}

# Enable autodoc debugging and improve import handling
autodoc_mock_imports_debug = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_inherit_docstrings = True

# Suppress warnings for missing references
nitpicky = False
nitpick_ignore = [
    ("py:class", "sklearn.base.BaseEstimator"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "pandas.DataFrame"),
    ("py:obj", "sklearn.base.BaseEstimator"),
]

# Add better link checking
linkcheck_ignore = [
    r'http://localhost:\d+/',
    r'https://example\.com',
]

autodoc_mock_imports = [
    "numpy",
    "pandas", 
    "scipy",
    "sklearn",
    "sklearn.base",
    "sklearn.preprocessing",
    "sklearn.linear_model",
    "sklearn.tree",
    "sklearn.utils",
    "sklearn.utils.validation",
    "sklearn.metrics",
    "sklearn.model_selection",
    "sklearn.datasets",
    "sklearn.ensemble",
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
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

# Intersphinx configuration
intersphinx_timeout = 10
intersphinx_retries = 3
intersphinx_disabled_reftypes = ["*"]  # Disable intersphinx for problematic references

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

# -- NBSphinx configuration --------------------------------------------------

# Execute notebooks during build
nbsphinx_execute = "always"
nbsphinx_allow_errors = True
nbsphinx_timeout = 600  # 10 minutes

# Configure execution environment
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'png', 'svg'}",
    "--InlineBackend.rc={'figure.dpi': 150}",
]

# Kernel and execution settings
nbsphinx_kernel_name = 'python3'
nbsphinx_codecell_lexer = 'ipython3'

# Prevent prompt numbers in output
nbsphinx_prompt_width = "0"


# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = f"{project} v{release}"
html_static_path = ["_static"]

# GitHub Pages configuration
html_baseurl = "https://finite-sample.github.io/stable-cart/"
html_use_smartypants = False
html_copy_source = False
html_show_sourcelink = False

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
    "announcement": "<strong>🎯 Stable-CART:</strong> Individual decision trees with stability-focused modifications.",
}

# Add custom CSS
html_css_files = [
    "custom.css",
]
