# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-2-Clause
"""Configuration for Sphinx"""

project = "sphinx-rst-builder"
copyright = "2019, David Fritzsche"
author = "David Fritzsche"
needs_sphinx = '2.0'

release = "0.0.1"
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon', # use Google style docstrings
    'sphinx.ext.viewcode', # link to source code
    'sphinx_rst_builder',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

# The master toctree document.
master_doc = 'index'


def setup(app):
    app.add_object_type('confval', 'confval',
                        objname='configuration value',
                        indextemplate='pair: %s; configuration value')
