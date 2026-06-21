"""Sphinx configuration for the fortranspire documentation."""
from __future__ import annotations

import os
import sys
from datetime import date

# Make the package importable for autodoc when building on RTD.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "fortranspire"
author = "Loïc Maurin"
copyright = f"{date.today().year}, {author}"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("fortranspire")
except Exception:  # pragma: no cover — fallback for unbuilt envs
    release = "0.1.0"

version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "linkify",
    "substitution",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc / autosummary
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "loki",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_mistralai",
    "langchain_openai",
    "langgraph",
    "fastmcp",
    "jax",
    "flax",
    "equinox",
    "Cython",
    "numpy",
]

# Cross-project references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"fortranspire {release}"
html_theme_options = {
    "source_repository": "https://github.com/maurinl26/fortranspire",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Misc --------------------------------------------------------------------

nitpicky = False
