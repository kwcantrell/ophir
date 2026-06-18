"""Sphinx configuration for the ophir documentation.

The package lives under ``../src`` (src layout). All heavy / optional / and
side-effecting third-party imports are mocked via ``autodoc_mock_imports`` so
that ``sphinx-build`` works without torch, gradio, etc. installed and without
triggering import-time side effects (e.g. ``register.py`` and ``models.py``).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# conf.py lives in docs/; the importable package is ../src/ophir.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "ophir"
author = "Kalen Cantrell"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.6.1"
version = "0.6.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autosummary / autodoc ---------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Mock every heavy / optional / side-effecting import so the docs build needs
# none of the runtime dependencies. ``typer`` is intentionally NOT mocked so
# that ``ophir.cli``'s command functions document correctly.
autodoc_mock_imports = [
    "torch",
    "lightning",
    "transformers",
    "gradio",
    "langchain",
    "langchain_core",
    "langchain_ollama",
    "massive",
    "yfinance",
    "plotly",
    "tensorboard",
    "tensorboardX",
    "ipywidgets",
    "tqdm",
    "dotenv",
    "pyarrow",
    "numpy",
    "pandas",
]

# -- Napoleon (Google + NumPy docstrings) ------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
# Render "Attributes" entries as inline :ivar: fields rather than separate
# attribute directives, so they do not collide with autodoc's own
# documentation of the (dataclass) fields.
napoleon_use_ivar = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "ophir"
# Canonical URL of the GitHub Pages site (used for sitemap / canonical links).
html_baseurl = "https://kwcantrell.github.io/ophir/"
