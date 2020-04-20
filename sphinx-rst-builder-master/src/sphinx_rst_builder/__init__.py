# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-2-Clause
"""Sphinx extension to output reST files.

"""

from sphinx.builders import Builder
from sphinx.writers.text import STDINDENT

from ._builder import RstBuilder
from ._writer import RstWriter



def setup(app):
    """Setup this extension"""
    app.require_sphinx('2.0')
    app.add_builder(RstBuilder)
    app.add_config_value('rst_file_suffix', ".rst", False)
    """This is the file name suffix for reST files"""
    app.add_config_value('rst_link_suffix', None, False)
    """The is the suffix used in internal links. By default, takes the same value as rst_file_suffix"""
    app.add_config_value('rst_file_transform', None, False)
    """Function to translate a docname to a filename. By default, returns docname + rst_file_suffix."""
    app.add_config_value('rst_link_transform', None, False)
    """Function to translate a docname to a (partial) URI. By default, returns docname + rst_link_suffix."""
    app.add_config_value('rst_indent', STDINDENT, False)
