#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/ngrion/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute and install Apode
"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

from setuptools import setup

from ez_setup import use_setuptools
use_setuptools()


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = ["numpy", "scipy", "attrs", "matplotlib"]

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

with open(PATH / "README.md") as fp:
    LONG_DESCRIPTION = fp.read()

with open(PATH / "apode" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', '').strip()
            break


DESCRIPTION = "Poverty and Inequality Analysis in Python"


# =============================================================================
# FUNCTIONS
# =============================================================================

def do_setup():
    setup(
        name="apode",
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',

        author=[
            "Néstor Grión",
            "Sofia Sappia"],
        author_email="ngrion@gmail.com",
        url="https://github.com/ngrion/apode",
        license="MIT",

        keywords=[
            "apode", "poverty", "inequality", "concentration",
            "welfare", "polarization"],

        classifiers=[
            "Development Status :: 1 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering"],

        packages=["apode"],
        py_modules=["ez_setup"],

        install_requires=REQUIREMENTS)


if __name__ == "__main__":
    do_setup()
