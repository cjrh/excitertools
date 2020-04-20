
Sphinx reStructuredText Builder
*******************************

`Sphinx <http://sphinx-doc.org/>`_ extension to build reST (`reStructuredText <http://docutils.sourceforge.net/rst.html>`_) files.

This extension is in particular useful to use in combination with the autodoc
extension to automatically generate documentation for use by any rst parser
(such as the GitHub wiki).

In itself, the extension is fairly straightforward – it takes the parsed reST
file from `Sphinx <http://sphinx-doc.org/>`_ and outputs it as reST.


Requirements
============

* `Sphinx <http://sphinx-doc.org/>`_ 2.0 or later

* Python 3.6 or later


Installing
==========


Using pip
---------

..

   pip install sphinx-rst-builder


Manual
------

..

   git clone `https://github.com/davidfritzsche/sphinx-rst-builder.git <https://github.com/davidfritzsche/sphinx-rst-builder.git>`_
   cd sphinx-rst-builder
   python setup.py install

If you want to take a look and have a try, you can put the reST builder in
an extension subdirectory, and adjust ``sys.path`` to tell Sphinx where to
look for it:

* Add the extensions directory to the path in ``conf.py``. E.g.

  ..
     sys.path.append(os.path.abspath(‘exts’))


Usage
=====

* Set the builder as a extension in ``conf.py``:

  ..
     extensions = [‘sphinx_rst_builder’]

* Run sphinx-build with target ``rst``:

  ..
     sphinx-build -b rst -c . build/rst


Configuration
=============

The following four configuration variables are defined by sphinxcontrib.restbuilder:

``rst_file_suffix``

   This is the file name suffix for generated reST files.  The default is
   ``".rst"``.

``rst_link_suffix``

   Suffix for generated links to reST files.  The default is whatever
   ``rst_file_suffix`` is set to.

``rst_file_transform``

   Function to translate a docname to a filename.
   By default, returns *docname* + ``rst_file_suffix``.

``rst_link_transform``

   Function to translate a docname to a (partial) URI.
   By default, returns *docname* + ``rst_link_suffix``.


Further Reading
===============

* `Sphinx <http://sphinx-doc.org/>`_

* `reStructuredText <http://docutils.sourceforge.net/rst.html>`_


Feedback
========

The reST builder is in a preliminary state. It’s not (yet) widely used, so
any feedback is particularly welcome.
