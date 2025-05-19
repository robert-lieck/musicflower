# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# for Plotly in Sphinx gallery examples
import plotly.io as pio
pio.renderers.default = 'sphinx_gallery'


# -- Project information -----------------------------------------------------

project = 'MusicFlower'
copyright = '2022, Robert Lieck'
author = 'Robert Lieck'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'autoclasstoc',
    'sphinx.ext.imgmath',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'style_nav_header_background': 'black',
}
html_context = {
    "display_github": True,
    "github_user": "robert-lieck",
    "github_repo": "musicflower",
    "github_version": "main",
    "conf_py_path": "/doc/",
}
html_logo = 'logo_96.png'
# don't show the "View page source" link in the RTD theme
html_show_sourcelink = False
# use svg in imgmath extension
imgmath_image_format='svg'
# intersphinx mappings
intersphinx_mapping = {
    'triangularmap': ('https://robert-lieck.github.io/triangularmap', None),
    'pitchscapes': ('https://robert-lieck.github.io/pitchscapes', None),
}

# include all Python files in example gallery (not just files starting with "plot_" as the default)
sphinx_gallery_conf = {"filename_pattern": ''}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_extra_path = ['extra']
