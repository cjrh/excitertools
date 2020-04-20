#!/bin/sh

sphinx-build -q -a -E -b rst -d build/readme docs/readme .
sphinx-build -q -a -E -b rst -d build/changelog docs/changelog .
